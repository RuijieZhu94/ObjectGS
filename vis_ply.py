import struct
import numpy as np
import cv2
from collections import Counter
from collections import defaultdict
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
import os

class ID2RGBConverter:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.color_map = self._generate_color_map()

    def _generate_color_map(self):
        # Generate 256 random RGB colors, each color is a tuple (0-255 range)
        return np.random.randint(0, 256, size=(256, 3), dtype=np.uint8)

    def convert(self, obj: int):
        if obj == 0:
            return 0, np.array([0, 0, 0], dtype=np.uint8)  # Predefine class 0 as black
        if 0 <= obj <= 255:
            return obj, self.color_map[obj]  # Get color from the fixed color map
        else:
            raise ValueError("ID out of range, should be between 0 and 255")


def fetchPly(path):
    print("read ply file from {}".format(path))
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])

    label_ids = np.array(vertices['label']).T

    return BasicPointCloud(points=positions, colors=colors, normals=normals, label_ids=label_ids)


# Save point cloud to .ply file
def storePly(path, xyz, rgb, label):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('label', 'u1')]
    
    # Create an array for normals (set to 0 here)
    normals = np.zeros_like(xyz)
    label = label.reshape(-1, 1)

    # Combine xyz, normals, and rgb data
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, label), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create a PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def main():

    ply_path = "datasets/lerf_mask/figurines/sparse/0/points3D_prob.ply"
    output_ply_path = os.path.join(os.path.dirname(ply_path), "points3D_prob_color.ply")
    # Read point cloud data
    pcd = fetchPly(ply_path)

    # Randomly generate a color converter
    converter = ID2RGBConverter()
    # print(np.unique(pcd.label_ids))
    # Iterate through all label IDs and generate random colors for each ID
    for label_id in np.unique(pcd.label_ids):
        _, rgb = converter.convert(int(label_id))
        pcd.colors[pcd.label_ids == label_id] = rgb

    non_zero_points_mask = pcd.label_ids != 0
    filtered_pcd = type(pcd)(
        points=pcd.points[non_zero_points_mask],
        colors=pcd.colors[non_zero_points_mask],
        normals=pcd.normals[non_zero_points_mask],
        label_ids=pcd.label_ids[non_zero_points_mask]
    )
    storePly(output_ply_path, filtered_pcd.points, filtered_pcd.colors, filtered_pcd.label_ids)

    # Save point cloud to .ply file
    # storePly(output_ply_path, pcd.points, pcd.colors, pcd.label_ids)

    print("Point cloud saved as {}".format(output_ply_path))

main()