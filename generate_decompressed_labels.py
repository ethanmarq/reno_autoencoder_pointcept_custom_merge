import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def transfer_labels(original_pc_path, decompressed_pc_path, original_label_path, new_label_path):
    """
    Transfers labels from an original point cloud to a decompressed one via KNN.
    """
    # 1. Load original and decompressed point clouds
    original_pc = o3d.io.read_point_cloud(original_pc_path)
    original_points = np.asarray(original_pc.points)
    
    decompressed_pc = o3d.io.read_point_cloud(decompressed_pc_path)
    decompressed_points = np.asarray(decompressed_pc.points)
    
    # 2. Load original labels
    original_labels = np.fromfile(original_label_path, dtype=np.int32)
    
    if original_points.shape[0] != original_labels.shape[0]:
        raise ValueError("Original points and labels must have the same quantity.")

    # 3. Build a KD-Tree on the original points for fast nearest-neighbor search
    kdtree = cKDTree(original_points)
    
    # 4. For each decompressed point, find the index of the nearest original point
    distances, indices = kdtree.query(decompressed_points, k=1)
    
    # 5. Create the new label array by taking the labels from the found indices
    new_labels = original_labels[indices]
    
    # 6. Save the new labels
    new_labels.tofile(new_label_path)
    print(f"Saved new labels to {new_label_path}")

# You would then write a script to loop this function over your entire dataset.