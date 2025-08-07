import os
import numpy as np
import glob
from collections.abc import Sequence
from scipy.spatial import cKDTree

from .builder import DATASETS
from .defaults import DefaultDataset
from ..models.backbones.reno.decompress_tensor import RENO_Decompressor 

@DATASETS.register_module()
class CompressedGOOSEDataset(DefaultDataset):
    def __init__(self,
                 ignore_index=-1,
                 original_data_root='/home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/Pointcept/data/compressed_goose',
                 ckpt_path='/home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/Pointcept/checkpoints/reno/ckpt.pt',
                 channels=32,
                 kernel_size=3,
                 device='cuda:0',
                 **kwargs):
        
        self.ckpt_path = ckpt_path
        self.channels = channels
        self.kernel_size = kernel_size
        self.device = device
        
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        self.original_data_root = original_data_root
        
        self.decompressor = None
        
        
        
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        """
        Finds the list of compressed files.
        Assumes compressed files are in a 'compressed_lidar' directory.
        """
            
        if isinstance(self.split, str):
            data_list = glob.glob(
                os.path.join(self.data_root, "compressed_lidar", self.split, "*/*.bin")
            )
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(
                    os.path.join(self.data_root, "compressed_lidar", split, "*/*.bin")
                )
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        """
        Decompresses a point cloud and assigns labels via nearest neighbor search.
        """
        if self.decompressor is None:
            self.decompressor = RENO_Decompressor(
                ckpt_path=self.ckpt_path,
                channels=self.channels,
                kernel_size=self.kernel_size,
                device=self.device
            )
        
        # 1. Get path to the compressed file and decompress it
        compressed_path = self.data_list[idx % len(self.data_list)]
        sparse_tensor = self.decompressor.decompress_file(compressed_path)

        decompressed_coords = sparse_tensor.coords[:, 1:].cpu().numpy()
        decompressed_feats = sparse_tensor.feats.cpu().numpy()

        # 2. Get path to the original uncompressed file and labels
        # This assumes a consistent file structure between compressed and original data.
        # Example: data_root/compressed_lidar/train/xyz.bin -> original_data_root/lidar/train/xyz.bin
        relative_path = os.path.relpath(compressed_path, os.path.join(self.data_root, "compressed_lidar"))
        original_data_path = os.path.join(self.original_data_root, "lidar", relative_path)
        
        label_path = original_data_path.replace("lidar", "labels_challenge").replace(".bin", ".label").replace("vls128", "goose").replace("_pcl.", "_goose.")

        # 3. Load original data and labels to build a nearest neighbor tree
        with open(original_data_path, "rb") as b:
            original_scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        
        original_coords = original_scan[:, :3]

        if os.path.exists(label_path):
            with open(label_path, "rb") as a:
                original_segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                original_segment = original_segment & 0x0000FFFF
                if np.any(original_segment > 8):
                    print(f"The following label file contains more than 8 classes so its probably the wrong version: {label_path}")
                    original_segment[original_segment > 8] = 0
        else:
            original_segment = np.zeros(original_scan.shape[0]).astype(np.int32)

        # 4. Use cKDTree for efficient nearest neighbor search
        tree = cKDTree(original_coords)
        _, indices = tree.query(decompressed_coords)
        
        # 5. Assign the original labels to the decompressed points
        segment = original_segment[indices]
        segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(np.int32)
        
        # 6. Prepare the final data dictionary
        data_dict = dict(
            coord=decompressed_coords,
            feat=decompressed_feats,  # Using the RENO features as the new `feat`
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        data_name = f"{file_name}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        # ... (Same as the original GOOSEDataset)
        learning_map = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 0,
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        # ... (Same as the original GOOSEDataset)
        learning_map_inv = {
            ignore_index: ignore_index,
            0 : 0,
            1 : 1,
            2 : 2,
            3 : 3,
            4 : 4,
            5 : 5,
            6 : 6,
            7 : 7,
        }
        return learning_map_inv