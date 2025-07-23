import torch
import torch.nn as nn
import os

# Pointcept-specific imports for building models and criteria
from .builder import MODELS, build_backbone, build_criterion
from ..structures import Point

# RENO-specific imports (assuming you placed them correctly)
from ..backbones.reno.network import Network as RenoDecoder
from torchsparse import SparseTensor

@MODELS.register_module("MergedSegmentor")
class MergedSegmentor(nn.Module):
    def __init__(
        self,
        backbone=None,          # Config dict for PTv3
        criteria=None,          # List of loss configs
        num_classes=8,
        backbone_out_channels=64,
        reno_ckpt=None,         # Path to RENO checkpoint
        ptv3_ckpt=None,         # Path to PTv3 checkpoint
        recon_loss_weight=0.1,  # Weight for the reconstruction loss
    ):
        super().__init__()
        # ======================= 1. BUILD MODELS =======================
        # Build the RENO decoder
        # These parameters are from the original RENO architecture
        self.decoder = RenoDecoder(channels=32, kernel_size=3)
        
        # Build the PTv3 backbone using the provided config dict
        self.segmentation_backbone = build_backbone(backbone)
        
        # Build the final linear layer for segmentation predictions
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        
        # Build the loss functions from the config
        self.criteria = build_criterion(criteria)
        self.recon_loss_weight = recon_loss_weight

        # ======================= 2. LOAD CHECKPOINTS =======================
        print("Initializing MergedSegmentor...")
        if reno_ckpt and os.path.exists(reno_ckpt):
            print(f"Loading RENO checkpoint from: {reno_ckpt}")
            self.decoder.load_state_dict(torch.load(reno_ckpt, map_location='cpu'))
        else:
            print("WARNING: RENO checkpoint not found. Using random RENO weights.")

        if ptv3_ckpt and os.path.exists(ptv3_ckpt):
            print(f"Loading PTv3 checkpoint from: {ptv3_ckpt}")
            state_dict = torch.load(ptv3_ckpt, map_location='cpu')
            # Pointcept saves optimizer state, etc. The model weights are in the "model" key.
            # strict=False ignores keys that don't match (like a different seg_head)
            self.segmentation_backbone.load_state_dict(state_dict["model"], strict=False)
        else:
            print("WARNING: PTv3 checkpoint not found. Using random PTv3 weights.")

    def forward(self, input_dict):
        # The input_dict contains the original, uncompressed data from the loader
        # We will simulate the full compress -> decompress -> segment pipeline here
        
        # ======================= STAGE 1: RENO PIPELINE =======================
        # This part simulates RENO's process to get both a reconstruction loss
        # and the features needed for segmentation.
        
        # Create the initial sparse tensor for RENO from the input data
        # Note: RENO expects integer coordinates
        quantized_coord = (input_dict["coord"] / 0.05).round().int() # Assuming 5cm voxel size
        reno_input = SparseTensor(
            coords=torch.cat([input_dict["offset"].unsqueeze(-1), quantized_coord], dim=1),
            feats=torch.ones(quantized_coord.shape[0], 1, device=quantized_coord.device)
        )
        
        # Calculate the reconstruction loss (rate-distortion) using RENO's forward pass
        # This is the loss for the compression task
        loss_reconstruction = self.decoder(reno_input)
        
        # Now, get the features for the segmentation backbone.
        # This involves running the feature-generation part of the RENO decoder.
        # This logic is adapted from the RENO training script.
        with torch.no_grad(): # We don't need gradients for this part of the simulation
            data_ls = []
            x = reno_input
            while True:
                x = self.decoder.fog(x)
                data_ls.append((x.coords.clone(), x.feats.clone()))
                if x.coords.shape[0] < 64: break
            data_ls = data_ls[::-1]

        # Use the RENO decoder layers to generate features from the (perfectly known) occupancy codes
        # This simulates decoding and provides the features for PTv3
        x_C, x_O = data_ls[0]
        x_F = self.decoder.prior_embedding(x_O.int()).view(-1, self.decoder.channels)
        x = SparseTensor(coords=x_C, feats=x_F)
        x = self.decoder.prior_resnet(x)
        
        # TODO: This is a simplification. A full implementation would loop through all scales
        # to generate the final high-resolution feature tensor, just like in RENO's code.
        # For a starter, using the features from the first upsampling pass is sufficient.
        x_up_C, x_up_F = self.decoder.fcg(x.coords, x.feats.int(), x.feats)
        x_up_F = self.decoder.target_embedding(x_up_F, x_up_C)
        decoded_features_sparse = SparseTensor(coords=x_up_C, feats=x_up_F)

        '''

        base_x_coords = torch.tensor(base_x_coords.reshape(-1, 3), device=device) 
        base_x_feats = torch.tensor(base_x_feats.reshape(-1, 1), device=device)

        x = SparseTensor(coords=torch.cat((base_x_feats*0, base_x_coords), dim=-1), feats=base_x_feats).cuda()
        byte_stream_ls = op.unpack_byte_stream(byte_stream)

        for byte_stream_idx in range(0, len(byte_stream_ls), 2):
            byte_stream_s0 = byte_stream_ls[byte_stream_idx]
            byte_stream_s1 = byte_stream_ls[byte_stream_idx+1]

            # embedding prior scale feats
            x_O = x.feats.int()
            x.feats = net.prior_embedding(x_O).view(-1, net.channels) # (N_d, C)
            x = net.prior_resnet(x) # (N_d, C)

            # target embedding
            x_up_C, x_up_F = net.fcg(x.coords, x_O, x_F=x.feats)
            x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)

            x_up_F = net.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = net.target_resnet(x_up)


            x_up_O_prob_s0 = net.pred_head_s0(x_up.feats) # (B*Nt, 16)
            x_up_O_cdf_s0 = torch.cat((x_up_O_prob_s0[:, 0:1]*0, x_up_O_prob_s0.cumsum(dim=-1)), dim=-1) # (B*Nt, 16)
            x_up_O_cdf_s0 = torch.clamp(x_up_O_cdf_s0, min=0, max=1)
            x_up_O_cdf_s0_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s0, True)
            x_up_O_cdf_s0_norm = x_up_O_cdf_s0_norm.cpu()
            x_up_O_s0 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s0_norm, byte_stream_s0).cuda()

            x_up_O_prob_s1 = net.pred_head_s1(x_up.feats + net.pred_head_s1_emb(x_up_O_s0.long())) # (B*Nt, 16)
            x_up_O_cdf_s1 = torch.cat((x_up_O_prob_s1[:, 0:1]*0, x_up_O_prob_s1.cumsum(dim=-1)), dim=-1) # (B*Nt, 16)
            x_up_O_cdf_s1 = torch.clamp(x_up_O_cdf_s1, min=0, max=1)
            x_up_O_cdf_s1_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s1, True)
            x_up_O_cdf_s1_norm = x_up_O_cdf_s1_norm.cpu()
            x_up_O_s1 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s1_norm, byte_stream_s1).cuda()

            x_up_O = x_up_O_s1 * 16 + x_up_O_s0

            x = SparseTensor(coords=x_up_C, feats=x_up_O.unsqueeze(-1)).cuda()
            
        # decode the last layer
        scan = net.fcg(x.C, x.F)


        '''

        # ======================= STAGE 2: PTv3 SEGMENTATION =======================
        # Create a new Point object for the PTv3 backbone, using the decoded features
        ptv3_input_dict = {
            "coord": decoded_features_sparse.C,
            "feat": decoded_features_sparse.F,
            "grid_coord": decoded_features_sparse.C[:, 1:], # Or pass through GridSample again
            "offset": input_dict["offset"] # Pass along the original offset
        }
        point = Point(ptv3_input_dict)
        
        # Run the PTv3 backbone
        point = self.segmentation_backbone(point)
        # Run the segmentation head
        seg_logits = self.seg_head(point.feat)
        
        # ======================= STAGE 3: COMBINED LOSS =======================
        if self.training:
            # Calculate the segmentation loss
            loss_seg = self.criteria(seg_logits, input_dict["segment"])
            
            # Combine the losses
            total_loss = loss_seg + self.recon_loss_weight * loss_reconstruction
            return dict(loss=total_loss, loss_seg=loss_seg, loss_recon=loss_reconstruction)
        
        # For evaluation/testing, just return the segmentation predictions
        else:
            loss_seg = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss_seg, seg_logits=seg_logits)