"""
GOOSE Dataset

Author: Raphael Hagmanns (hagmanns@kit.edu)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import glob
from collections.abc import Sequence

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class GOOSEDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
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

                #map all ints > 8 in segment to 0 and print the label file if such a case appears:
                if np.any(segment > 8):
                    print(f"The following label file contains more than 8 classes so its probably the wrong version: {label_file}")
                    segment[segment > 8] = 0

                segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        data_dict = dict(
            coord=coord,
            strength=strength,
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
