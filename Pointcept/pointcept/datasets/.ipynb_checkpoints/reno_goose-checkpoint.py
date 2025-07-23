# pointcept/datasets/reno_goose.py
import os
import numpy as np
import glob
import torch
from collections.abc import Sequence

# Pointcept and RENO imports
from .builder import DATASETS
from .defaults import DefaultDataset
from torchsparse.utils.collate import sparse_collate_fn as reno_collate
from torch.utils.data.dataloader import default_collate

@DATASETS.register_module("RenoGooseDataset")
class RenoGooseDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, **kwargs):
        # This init is from your original GOOSEDataset, which is correct.
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        # This is the corrected file-finding logic from your original dataset.
        if isinstance(self.split, str):
            data_list = glob.glob(
                os.path.join(self.data_root, "lidar", self.split, "*/*.bin")
            )
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(
                    os.path.join(self.data_root, "lidar", split, "*/*.bin")
                )
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        # This is also from your original dataset. It correctly loads the
        # raw coordinates, strength, and segmentation labels.
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("lidar", "labels_challenge").replace(".bin", ".label").replace("vls128", "goose").replace("_pcl.","_goose.")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = segment & 0x0000FFFF
                if np.any(segment > 8):
                    print(f"Warning: Label file contains more than 8 classes: {label_file}")
                    segment[segment > 8] = 0
                segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        
        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
        )
        return data_dict

    def __getitem__(self, idx):
        # This is the new part. It calls the original get_data and then
        # applies the RENO-specific processing.
        data_dict = self.get_data(idx)
        
        # Perform RENO's preprocessing on the coordinates
        coord_tensor = torch.tensor(data_dict["coord"], dtype=torch.float)
        # RENO's quantization: offset and scale. Using posQ=16 as an example.
        quantized_coord = torch.round((coord_tensor / 0.001 + 131072) / 16).int()
        
        # Ensure segment is a tensor
        segment_tensor = torch.tensor(data_dict["segment"], dtype=torch.long)
        
        # Return the dictionary expected by our custom reno_sparse_collate_fn
        return dict(
            coord=quantized_coord,
            segment=segment_tensor
        )
        
    @staticmethod
    def get_learning_map(ignore_index):
        # From your original GOOSEDataset
        learning_map = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 0
        }
        return learning_map
    
    


from torchsparse.utils.collate import sparse_collate
from torch.utils.data.dataloader import default_collate

def reno_sparse_collate_fn(batch):
    # Prepare data for RENO's official collate function
    reno_data_list = [{'coord': data['coord'], 'feat': torch.ones(len(data['coord']), 1)} for data in batch]
    # Use torchsparse's collate function. It returns a dictionary {'input': SparseTensor}.
    reno_batched = reno_collate(reno_data_list)

    # Separately batch the segmentation labels
    segment_batch = default_collate([data['segment'] for data in batch])

    return {
        "reno_input": reno_batched, # Extract the SparseTensor
        "segment": segment_batch
    }