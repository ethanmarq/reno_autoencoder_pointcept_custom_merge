import os
import numpy as np
import glob
import torch
from collections.abc import Sequence

# Pointcept and RENO/torchsparse imports
from .builder import DATASETS
from .defaults import DefaultDataset
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn as reno_collate
from torch.utils.data.dataloader import default_collate

@DATASETS.register_module("RenoGooseDataset")
class RenoGooseDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, **kwargs):
        # This init is from your original GOOSEDataset, which is correct.
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
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
        data_dict = self.get_data(idx)
        coord = torch.tensor(data_dict["coord"][:, :3], dtype=torch.float)
        quantized_coord = torch.round((coord / 0.001 + 131072) / 16).int()
        
        # Create the SparseTensor here for a SINGLE sample.
        # The collate function will combine these into a batch.
        reno_input_tensor = SparseTensor(
            coords=quantized_coord,
            feats=torch.ones(quantized_coord.shape[0], 1)
        )
        
        segment_tensor = torch.tensor(data_dict["segment"], dtype=torch.long)
        
        # Return a dictionary with the key 'input', just like RENO's original dataset
        return {"input": reno_input_tensor, "segment": segment_tensor}
    
    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 0, # Map sky to unknown
        }
        return learning_map
    
    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
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


# This function correctly batches the SparseTensors created by the dataset
def reno_sparse_collate_fn(batch):
    # `batch` is a list of dicts: [{'input': SparseTensor, 'segment': Tensor}, ...]
    collated_dict = reno_collate(batch)
    collated_dict['reno_input'] = collated_dict.pop('input')
    return collated_dict

'''
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
from torchsparse import SparseTensor

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
        data_dict = self.get_data(idx)
        
        coord_xyz = data_dict["coord"][:, :3]
        
        coord_tensor = torch.tensor(coord_xyz, dtype=torch.float)
        # Apply RENO's quantization
        quantized_coord = torch.round((coord_tensor / 0.001 + 131072) / 16).int()
        
        # Create the SparseTensor here, with a batch index of 0 for a single sample.
        num_points = quantized_coord.shape[0]
        # Note: The batch index is a placeholder; the collate function will fix it.
        batch_indices = torch.zeros(num_points, 1, dtype=torch.int)
        sparse_coords = torch.cat([batch_indices, quantized_coord], dim=1)
        
        reno_input_tensor = SparseTensor(
            coords=sparse_coords,
            feats=torch.ones(num_points, 1, dtype=torch.float)
        )
        
        segment_tensor = torch.tensor(data_dict["segment"], dtype=torch.long)
        
        # Return a dictionary with the key 'input', just like RENO's original dataset
        return {
            "input": reno_input_tensor,
            "segment": segment_tensor
        }
        
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
    # `batch` is a list of dicts: [{'input': SparseTensor, 'segment': Tensor}, ...]
    
    # Use RENO's collate function. It knows how to find the 'input' key
    # and correctly batch the SparseTensors. It will also handle 'segment'.
    collated_dict = reno_collate(batch)
    
    # Rename the key to match what our MergedSegmentor expects.
    collated_dict['reno_input'] = collated_dict.pop('input')
    
    return collated_dict
'''