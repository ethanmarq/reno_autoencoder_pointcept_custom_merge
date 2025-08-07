import os
import time
import random
import numpy as np
import torch
import torchac
import io
from tqdm import tqdm
from glob import glob
from torchsparse import SparseTensor
from torchsparse.nn import functional as F

from pointcept.models.backbones.reno.network import Network as RENO_Network
import pointcept.models.backbones.reno.kit.io as RENO_IO
import pointcept.models.backbones.reno.kit.op as RENO_OP

class RENO_Decompressor:
    def __init__(self, ckpt_path: str, channels: int = 32, kernel_size: int = 3, device: str = 'cuda'):
        random.seed(1)
        np.random.seed(1)
        self.device = device

        conv_config = F.conv_config.get_default_conv_config()
        conv_config.kmap_mode = "hashmap"
        F.conv_config.set_global_conv_config(conv_config)

        self.net = RENO_Network(channels=channels, kernel_size=kernel_size)
        self.net.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.net.to(device).eval()

        self._warmup()
        self.op = RENO_OP

    def _warmup(self):
        random_coords = torch.randint(low=0, high=4096, size=(4096, 3)).int().to(self.device)
        self.net(SparseTensor(coords=torch.cat((random_coords[:, 0:1] * 0, random_coords), dim=-1),
                              feats=torch.ones((4096, 1))).to(self.device))

    def decompress_from_bytes(self, compressed_data: bytes) -> SparseTensor:
        with torch.no_grad():
            f_buffer = io.BytesIO(compressed_data)

            posQ = np.frombuffer(f_buffer.read(2), dtype=np.float16)[0]
            base_x_len = np.frombuffer(f_buffer.read(4), dtype=np.int32)[0]
            base_x_coords = np.frombuffer(f_buffer.read(base_x_len * 4 * 3), dtype=np.int32)
            base_x_feats = np.frombuffer(f_buffer.read(base_x_len * 1), dtype=np.uint8)
            byte_stream = f_buffer.read()

            base_x_coords = torch.tensor(base_x_coords.reshape(-1, 3), device=self.device)
            base_x_feats = torch.tensor(base_x_feats.reshape(-1, 1), device=self.device)
            x = SparseTensor(coords=torch.cat((base_x_feats * 0, base_x_coords), dim=-1),
                             feats=base_x_feats).to(self.device)

            byte_stream_ls = self.op.unpack_byte_stream(byte_stream)

            for byte_stream_idx in range(0, len(byte_stream_ls), 2):
                byte_stream_s0 = byte_stream_ls[byte_stream_idx]
                byte_stream_s1 = byte_stream_ls[byte_stream_idx + 1]

                x_O = x.feats.int()
                x.feats = self.net.prior_embedding(x_O).view(-1, self.net.channels)
                x = self.net.prior_resnet(x)

                x_up_C, x_up_F = self.net.fcg(x.coords, x_O, x_F=x.feats)
                x_up_C, x_up_F = self.op.sort_CF(x_up_C, x_up_F)
                x_up_F = self.net.target_embedding(x_up_F, x_up_C)
                
                x_up = SparseTensor(coords=x_up_C, feats=x_up_F)

                x_up_O_prob_s0 = self.net.pred_head_s0(x_up.feats)
                x_up_O_cdf_s0 = torch.cat((x_up_O_prob_s0[:, 0:1] * 0, x_up_O_prob_s0.cumsum(dim=-1)), dim=-1)
                x_up_O_cdf_s0 = torch.clamp(x_up_O_cdf_s0, min=0, max=1)
                x_up_O_cdf_s0_norm = self.op._convert_to_int_and_normalize(x_up_O_cdf_s0, True)
                x_up_O_s0 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s0_norm.cpu(), byte_stream_s0).to(self.device)

                x_up_O_prob_s1 = self.net.pred_head_s1(x_up.feats + self.net.pred_head_s1_emb(x_up_O_s0.long()))
                x_up_O_cdf_s1 = torch.cat((x_up_O_prob_s1[:, 0:1] * 0, x_up_O_prob_s1.cumsum(dim=-1)), dim=-1)
                x_up_O_cdf_s1 = torch.clamp(x_up_O_cdf_s1, min=0, max=1)
                x_up_O_cdf_s1_norm = self.op._convert_to_int_and_normalize(x_up_O_cdf_s1, True)
                x_up_O_s1 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s1_norm.cpu(), byte_stream_s1).to(self.device)
                x_up_O = x_up_O_s1 * 16 + x_up_O_s0
                
                x = SparseTensor(coords=x_up_C, feats=x_up_O.unsqueeze(-1)).to(self.device)

            return x_up

    def decompress_file(self, file_path: str) -> SparseTensor:
        with open(file_path, 'rb') as f:
            compressed_data = f.read()
        return self.decompress_from_bytes(compressed_data)

    def decompress_files_and_save(self, input_glob, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        file_paths = glob(input_glob)
        dec_time_ls = []

        with torch.no_grad():
            for file_path in tqdm(file_paths):
                file_name = os.path.basename(file_path)
                decompressed_file_path = os.path.join(output_folder, file_name + '.ply')
                
                dec_time_start = time.time()
                # Decompress, but get the final scan from the last step for saving
                with open(file_path, 'rb') as f:
                    compressed_data = f.read()

                f_buffer = RENO_IO.BytesIO(compressed_data)
                posQ = np.frombuffer(f_buffer.read(2), dtype=np.float16)[0]
                base_x_len = np.frombuffer(f_buffer.read(4), dtype=np.int32)[0]
                base_x_coords = np.frombuffer(f_buffer.read(base_x_len * 4 * 3), dtype=np.int32)
                base_x_feats = np.frombuffer(f_buffer.read(base_x_len * 1), dtype=np.uint8)
                byte_stream = f_buffer.read()

                base_x_coords = torch.tensor(base_x_coords.reshape(-1, 3), device=self.device)
                base_x_feats = torch.tensor(base_x_feats.reshape(-1, 1), device=self.device)
                x = SparseTensor(coords=torch.cat((base_x_feats * 0, base_x_coords), dim=-1),
                                 feats=base_x_feats).to(self.device)
                byte_stream_ls = self.op.unpack_byte_stream(byte_stream)
                
                for byte_stream_idx in range(0, len(byte_stream_ls), 2):
                    byte_stream_s0 = byte_stream_ls[byte_stream_idx]
                    byte_stream_s1 = byte_stream_ls[byte_stream_idx + 1]
                    x_O = x.feats.int()
                    x.feats = self.net.prior_embedding(x_O).view(-1, self.net.channels)
                    x = self.net.prior_resnet(x)
                    x_up_C, x_up_F = self.net.fcg(x.coords, x_O, x_F=x.feats)
                    x_up_C, x_up_F = self.op.sort_CF(x_up_C, x_up_F)
                    x_up_F = self.net.target_embedding(x_up_F, x_up_C)
                    x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
                    x_up_O_prob_s0 = self.net.pred_head_s0(x_up.feats)
                    x_up_O_cdf_s0 = torch.cat((x_up_O_prob_s0[:, 0:1] * 0, x_up_O_prob_s0.cumsum(dim=-1)), dim=-1)
                    x_up_O_cdf_s0 = torch.clamp(x_up_O_cdf_s0, min=0, max=1)
                    x_up_O_cdf_s0_norm = self.op._convert_to_int_and_normalize(x_up_O_cdf_s0, True)
                    x_up_O_s0 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s0_norm.cpu(), byte_stream_s0).to(self.device)
                    x_up_O_prob_s1 = self.net.pred_head_s1(x_up.feats + self.net.pred_head_s1_emb(x_up_O_s0.long()))
                    x_up_O_cdf_s1 = torch.cat((x_up_O_prob_s1[:, 0:1] * 0, x_up_O_prob_s1.cumsum(dim=-1)), dim=-1)
                    x_up_O_cdf_s1 = torch.clamp(x_up_O_cdf_s1, min=0, max=1)
                    x_up_O_cdf_s1_norm = self.op._convert_to_int_and_normalize(x_up_O_cdf_s1, True)
                    x_up_O_s1 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s1_norm.cpu(), byte_stream_s1).to(self.device)
                    x_up_O = x_up_O_s1 * 16 + x_up_O_s0
                    x = SparseTensor(coords=x_up_C, feats=x_up_O.unsqueeze(-1)).to(self.device)

                scan = self.net.fcg(x.C, x.F)
                scan = (scan[:, 1:] * posQ - 131072) * 0.001

                dec_time_end = time.time()
                dec_time_ls.append(dec_time_end - dec_time_start)
                RENO_IO.save_ply_ascii_geo(scan.float().cpu().numpy(), decompressed_file_path)

        print('Total: {total_n:d} | Decode Time:{dec_time:.3f} | Max GPU Memory:{memory:.2f}MB'.format(
            total_n=len(dec_time_ls),
            dec_time=np.array(dec_time_ls).mean(),
            memory=torch.cuda.max_memory_allocated() / 1024 / 1024
        ))