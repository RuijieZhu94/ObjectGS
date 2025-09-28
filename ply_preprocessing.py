"""
Point Cloud Preprocessing for ObjectGS

This module provides functionality for preprocessing 3D point clouds with semantic labels
by projecting them onto 2D images and assigning colors/labels through various voting strategies.

Author: Ruijie Zhu
License: MIT
"""

import struct
import numpy as np
import cv2
from collections import Counter, defaultdict
from plyfile import PlyData, PlyElement
from scene.colmap_loader import (
    read_intrinsics_binary, read_extrinsics_binary, read_next_bytes, 
    read_intrinsics_text, read_extrinsics_text
)
import argparse
import os

def read_points3D_binary(path_to_model_file):
    """
    Parses COLMAP's points3D.bin file and returns a dictionary:
    point3D_id -> (x, y, z, r, g, b, error, track)
    where track is a list of (image_id, point2D_idx) tuples.
    """
    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    import struct
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        points3D = {}

        for _ in range(num_points):
            data = read_next_bytes(fid, 43, "QdddBBBd")  # point3D_id + xyz + rgb + error
            point3D_id = data[0]
            xyz = np.array(data[1:4])
            rgb = np.array(data[4:7])
            error = data[7]

            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)

            # track is a list of (image_id, point2D_idx)
            track = [(track_elems[i], track_elems[i + 1]) for i in range(0, len(track_elems), 2)]

            points3D[point3D_id] = (xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2], error, track)

    return points3D


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    point3D = {}

    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                point3D[count] = np.concatenate((xyz, rgb), axis=0)
                count += 1

    return point3D


class ID2RGBConverter:
    """Converter to map object IDs to unique RGB colors."""
    
    def __init__(self):
        self.all_id = []  # Store all generated IDs
        self.obj_to_id = {}  # Mapping from object ID to randomly generated color ID

    def _id_to_rgb(self, id: int):
        """Convert integer ID to RGB color."""
        rgb = np.zeros((3, ), dtype=np.uint8)  # Initialize RGB channels
        for i in range(3):
            rgb[i] = id % 256  # Take the lower 8 bits of the ID as the RGB channel value
            id = id // 256  # Right shift 8 bits to process the remaining part
        return rgb

    def convert(self, obj: int):
        """Convert single-channel ID to random RGB value."""
        if obj in self.obj_to_id:
            id = self.obj_to_id[obj]  # If the object already exists, get the corresponding ID
        else:
            # Randomly generate a unique ID and ensure no duplicates
            while True:
                id = np.random.randint(255, 256**3)
                if id not in self.all_id:
                    break
            self.obj_to_id[obj] = id  # Store the new ID in the dictionary
            self.all_id.append(id)  # Record this ID

        return id, self._id_to_rgb(id)  # Return the ID and corresponding RGB value

def corr_assign_final_colors(points3D, all_colors, all_labels):
    """Assign final colors using correlation-based voting.
    
    Args:
        points3D: Dictionary of 3D points
        all_colors: List of (point_id, color) tuples
        all_labels: List of (point_id, label) tuples
        
    Returns:
        Dictionary of updated 3D points with new colors and labels
    """
    from collections import defaultdict, Counter
    point_final_labels = {}    
    point_final_colors = {}
    
    colors_dict = defaultdict(list)
    labels_dict = defaultdict(list)

    for pid, color in all_colors:
        colors_dict[pid].append(color)

    for pid, label in all_labels:
        labels_dict[pid].append(label)

    for point_id in points3D:
        colors = colors_dict[point_id]
        labels = labels_dict[point_id]

        # Filter out items where label == 0 (background)
        filtered = [(c, l) for c, l in zip(colors, labels) if l != 0]
        if not filtered:
            continue  # Skip this point if there are no valid labels

        filtered_colors, filtered_labels = zip(*filtered)

        counter = Counter(filtered_labels)
        max_value = max(counter, key=counter.get)

        # Find the color corresponding to the most frequent label
        label_indices = [i for i, label in enumerate(filtered_labels) if label == max_value]
        max_color = filtered_colors[label_indices[0]]

        point_final_labels[point_id] = np.array(max_value)
        point_final_colors[point_id] = np.array(max_color)

    # Create a new points3D dictionary with updated colors and labels
    new_points3D = {}

    for point_id, point_data in points3D.items():
        # Original data format: (x, y, z, r, g, b, error, track)
        x, y, z, r, g, b, error, track = point_data
        r_new, g_new, b_new = point_final_colors.get(point_id, (r, g, b))
        label = point_final_labels.get(point_id, 0)

        new_points3D[point_id] = (x, y, z, r_new, g_new, b_new, label)

    return new_points3D

def majority_assign_final_colors(points3D, all_colors, all_labels):
    """Assign final colors using majority voting.
    
    Args:
        points3D: Dictionary of 3D points
        all_colors: List of (point_id, color) tuples
        all_labels: List of (point_id, label) tuples
        
    Returns:
        Dictionary of updated 3D points with final colors and labels
    """
    point_final_labels = {}    
    point_final_colors = {}
    
    colors_dict = defaultdict(list)
    labels_dict = defaultdict(list)

    # Build mapping dictionaries in advance to avoid lookups in loops
    for pid, color in all_colors:
        colors_dict[pid].append(color)

    for pid, label in all_labels:
        labels_dict[pid].append(label)

    # Iterate through points3D and find the most common labels and corresponding colors
    for point_id in points3D:
        colors = colors_dict[point_id]
        labels = labels_dict[point_id]
        
        if labels:
            # Use Counter to find the most frequent label
            counter = Counter(labels)
            max_value = max(counter, key=counter.get)
            
            # Find the color corresponding to the most frequent label
            label_indices = [i for i, label in enumerate(labels) if label == max_value]
            max_color = colors[label_indices[0]]  # Take the first matching color

            # Store results in final dictionaries
            point_final_labels[point_id] = np.array(max_value)
            point_final_colors[point_id] = np.array(max_color)
    
    # Update point cloud colors to final colors
    for point_id, color in point_final_colors.items():
        points3D[point_id][3:6] = color  # Update RGB colors

        # Update point cloud labels
        label = point_final_labels[point_id]
        
        # Add label as the 7th dimension
        points3D[point_id] = np.concatenate((points3D[point_id], [label]))

    return points3D

def prob_assign_final_colors(points3D, all_colors, all_labels):
    """Assign final colors using probability-based voting.
    
    Args:
        points3D: Dictionary of 3D points
        all_colors: List of (point_id, color) tuples
        all_labels: List of (point_id, label) tuples
        
    Returns:
        Dictionary of updated 3D points with sampled colors and labels
    """
    point_final_labels = {}    
    point_final_colors = {}
    
    colors_dict = defaultdict(list)
    labels_dict = defaultdict(list)

    # Build mapping dictionaries in advance to avoid lookups in loops
    for pid, color in all_colors:
        colors_dict[pid].append(color)

    for pid, label in all_labels:
        labels_dict[pid].append(label)

    # Iterate through points3D and vote based on probability distribution
    for point_id in points3D:
        colors = colors_dict[point_id]
        labels = labels_dict[point_id]

        if labels:
            # Use Counter to calculate label frequencies
            counter = Counter(labels)
            total = sum(counter.values())

            # Convert frequencies to probability distribution
            labels_list, counts = zip(*counter.items())
            probabilities = np.array(counts) / total

            # Randomly sample a label based on probabilities
            sampled_label = np.random.choice(labels_list, p=probabilities)

            # Find the color corresponding to the sampled label
            label_indices = [i for i, label in enumerate(labels) if label == sampled_label]
            sampled_color = colors[label_indices[0]]  # Take the first matching color

            # Store results in final dictionaries
            point_final_labels[point_id] = np.array(sampled_label)
            point_final_colors[point_id] = np.array(sampled_color)
    
    # Update point cloud colors to final colors
    for point_id, color in point_final_colors.items():
        points3D[point_id][3:6] = color  # Update RGB colors

        # Update point cloud labels
        label = point_final_labels[point_id]
        
        # Add label as the 7th dimension
        points3D[point_id] = np.concatenate((points3D[point_id], [label]))

    return points3D

def storePly(path, xyz, rgb, label):
    """Save point cloud data to PLY file format.
    
    Args:
        path: Output file path
        xyz: Point coordinates (N, 3)
        rgb: Point colors (N, 3)
        label: Point labels (N, 1)
    """
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('label', 'u1')]
    
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, label), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix.
    
    Args:
        qw, qx, qy, qz: Quaternion components
        
    Returns:
        3x3 rotation matrix
    """
    R = np.array([[1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
                  [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
                  [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]])
    return R


def _load_and_process_image(label_image_dir, color_image_dir, image_name, converter):
    """Load and process label and color images for a given view.
    
    Args:
        label_image_dir: Directory containing label images
        color_image_dir: Directory containing color images (can be None)
        image_name: Name of the image file
        converter: ID2RGBConverter instance
        
    Returns:
        Tuple of (color_image, label_image)
    """
    # Load label image
    label_image_file = os.path.join(label_image_dir, image_name)
    label_image_file = label_image_file.replace('.jpg', '.png').replace('.JPG', '.png')
    label_image = cv2.imread(label_image_file, -1)  # Load as single-channel grayscale
    
    # Load or generate color image
    if color_image_dir is not None:
        color_image_file = os.path.join(color_image_dir, image_name)
        color_image_file = color_image_file.replace('.jpg', '.png').replace('.JPG', '.png')
        color_image = cv2.imread(color_image_file, cv2.IMREAD_COLOR)
    else:
        # Generate color image from label image using converter
        color_image = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)
        for i in range(label_image.shape[0]):
            for j in range(label_image.shape[1]):
                obj_id = label_image[i, j]
                _, rgb_color = converter.convert(obj_id)
                color_image[i, j] = rgb_color
    
    return color_image, label_image


def _extract_camera_params(camera_data):
    """Extract camera parameters from COLMAP camera data.
    
    Args:
        camera_data: COLMAP camera data tuple
        
    Returns:
        Tuple of (fx, fy, cx, cy)
    """
    _, model_type, width, height, params = camera_data
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]  # Assuming pinhole model
    return fx, fy, cx, cy


def _extract_pose_params(image_data):
    """Extract pose parameters from COLMAP image data.
    
    Args:
        image_data: COLMAP image data tuple
        
    Returns:
        Tuple of (R, t) where R is rotation matrix and t is translation vector
    """
    _, qvec, tvec, camera_id, image_name, points2D, points3D_ids = image_data
    qw, qx, qy, qz = qvec
    tx, ty, tz = tvec
    
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    t = np.array([tx, ty, tz]).reshape((3, 1))
    
    return R, t

def project_points(points3D, R, t, fx, fy, cx, cy):
    """Project 3D points to 2D image plane.
    
    Args:
        points3D: Dictionary of 3D points
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        fx, fy: Focal lengths
        cx, cy: Principal point coordinates
        
    Returns:
        List of projected points (point_id, u, v)
    """
    projected_points = []
    for point_id, point_data in points3D.items():
        point = np.array([point_data[0], point_data[1], point_data[2]]).reshape(3, 1)
        # Transform world coordinates to camera coordinate system
        cam_point = np.dot(R, point) + t
        x, y, z = cam_point.flatten()
        # Project 3D point to 2D image using camera intrinsics
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        projected_points.append((point_id, int(u), int(v)))
    return projected_points

def get_point_colors_from_image(projected_points, color_image, label_image):
    """Get colors and labels for projected points from images.
    
    Args:
        projected_points: List of projected points (point_id, u, v)
        color_image: RGB color image
        label_image: Label image
        
    Returns:
        Tuple of (point_colors, point_labels) lists
    """
    point_colors = []
    point_labels = []
    for point_id, u, v in projected_points:
        if 0 <= u < color_image.shape[1] and 0 <= v < color_image.shape[0]:
            # Get RGB color value of the pixel
            color = color_image[v, u]  # Note: OpenCV loads images in (B, G, R) format
            point_colors.append((point_id, color))
            label = label_image[v, u]
            point_labels.append((point_id, label))
    return point_colors, point_labels


def majority_voting(images, points3D, cameras, label_image_dir, color_image_dir, converter, output_ply_path):
    """Perform majority voting to assign colors and labels to 3D points.
    
    Args:
        images: Dictionary of image data
        points3D: Dictionary of 3D points
        cameras: Dictionary of camera parameters
        label_image_dir: Directory containing label images
        color_image_dir: Directory containing color images (optional)
        converter: ID2RGBConverter instance
        output_ply_path: Output PLY file path
    """
    all_point_colors = []
    all_point_labels = []    
    # Iterate through all views
    for image_id, image_data in images.items():
        _, qvec, tvec, camera_id, image_name, points2D, points3D_ids = image_data
        
        # Extract pose and camera parameters
        R, t = _extract_pose_params(image_data)
        fx, fy, cx, cy = _extract_camera_params(cameras[camera_id])
        
        # Load and process images
        color_image, label_image = _load_and_process_image(
            label_image_dir, color_image_dir, image_name, converter
        )

        # Project point cloud to current view
        projected_points = project_points(points3D, R, t, fx, fy, cx, cy)

        # Get color for each point in this view
        point_colors, point_labels = get_point_colors_from_image(projected_points, color_image, label_image)

        # Save results
        all_point_colors.extend(point_colors)
        all_point_labels.extend(point_labels)

    # Calculate final colors for each point and update point cloud colors
    points3D = majority_assign_final_colors(points3D, all_point_colors, all_point_labels)

    # Extract point cloud coordinates and colors
    xyz = np.array([[point_data[0], point_data[1], point_data[2]] for point_data in points3D.values()])
    rgb = np.array([[point_data[3], point_data[4], point_data[5]] for point_data in points3D.values()])
    label = np.array([[point_data[6]] for point_data in points3D.values()])

    # Save point cloud as PLY file
    storePly(output_ply_path, xyz, rgb, label)

    print(f"Point cloud saved to {output_ply_path}")

def prob_voting(images, points3D, cameras, label_image_dir, color_image_dir, converter, output_ply_path):
    """Perform probability-based voting to assign colors and labels to 3D points.
    
    Args:
        images: Dictionary of image data
        points3D: Dictionary of 3D points
        cameras: Dictionary of camera parameters
        label_image_dir: Directory containing label images
        color_image_dir: Directory containing color images (optional)
        converter: ID2RGBConverter instance
        output_ply_path: Output PLY file path
    """
    all_point_colors = []
    all_point_labels = []    
    # Iterate through all views
    for image_id, image_data in images.items():
        _, qvec, tvec, camera_id, image_name, points2D, points3D_ids = image_data

        # Extract pose and camera parameters
        R, t = _extract_pose_params(image_data)
        fx, fy, cx, cy = _extract_camera_params(cameras[camera_id])
        
        # Load and process images
        color_image, label_image = _load_and_process_image(
            label_image_dir, color_image_dir, image_name, converter
        )

        # Project point cloud to current view
        projected_points = project_points(points3D, R, t, fx, fy, cx, cy)

        # Get color for each point in this view
        point_colors, point_labels = get_point_colors_from_image(projected_points, color_image, label_image)

        # Save results
        all_point_colors.extend(point_colors)
        all_point_labels.extend(point_labels)

    # Calculate final colors for each point using probability-based voting
    points3D = prob_assign_final_colors(points3D, all_point_colors, all_point_labels)

    # Extract point cloud coordinates and colors
    xyz = np.array([[point_data[0], point_data[1], point_data[2]] for point_data in points3D.values()])
    rgb = np.array([[point_data[3], point_data[4], point_data[5]] for point_data in points3D.values()])
    label = np.array([[point_data[6]] for point_data in points3D.values()])

    # Save point cloud as PLY file
    storePly(output_ply_path, xyz, rgb, label)

    print(f"Point cloud saved to {output_ply_path}")

def corr_voting(images, points3D, label_image_dir, converter, output_ply_path):
    """Perform correlation-based voting using track correspondence.
    
    Args:
        images: Dictionary of image data
        points3D: Dictionary of 3D points with track information
        label_image_dir: Directory containing label images
        converter: ID2RGBConverter instance
        output_ply_path: Output PLY file path
    """
    all_colors = []
    all_labels = []
    for point3D_id, point_data in points3D.items():
        x, y, z, r, g, b, error, track = point_data
        votes = []

        for image_id, point2D_idx in track:
            if image_id not in images:
                continue
            _, _, _, _, image_name, xys, _ = images[image_id]
            if point2D_idx >= len(xys):
                continue
            u, v = xys[point2D_idx]
            u = int(round(u))
            v = int(round(v))

            label_image_file = os.path.join(label_image_dir, image_name)
            label_image_file = label_image_file.replace('.jpg', '.png') if label_image_file.endswith('.jpg') else label_image_file.replace('.JPG', '.png')
            label_image = cv2.imread(label_image_file, -1)
            if label_image is None or v < 0 or v >= label_image.shape[0] or u < 0 or u >= label_image.shape[1]:
                continue

            obj_id = label_image[v, u]
            _, rgb_color = converter.convert(obj_id)

            all_colors.append((point3D_id, rgb_color))
            all_labels.append((point3D_id, obj_id))


    # Assign colors and labels through voting
    points3D = corr_assign_final_colors(points3D, all_colors, all_labels)

    # Extract and save
    xyz = np.array([[p[0], p[1], p[2]] for p in points3D.values()])
    rgb = np.array([[p[3], p[4], p[5]] for p in points3D.values()])
    label = np.array([[p[6]] for p in points3D.values()])

    storePly(output_ply_path, xyz, rgb, label)
    print(f"Point cloud saved to {output_ply_path}")


def main(args):
    """Main processing function.
    
    Args:
        args: Command line arguments containing dataset_path, algorithm, and output_ply_name
    """
    dataset_path = args.dataset_path

    for dataset_folder in os.listdir(dataset_path):
        print(f"Processing {dataset_folder}...")
        label_image_dir = os.path.join(dataset_path, dataset_folder, 'object_mask')
        color_image_dir = os.path.join(dataset_path, dataset_folder, 'color_mask')
        output_ply_path = os.path.join(dataset_path, dataset_folder, 'sparse/0/' + args.output_ply_name)

        # Try to load binary COLMAP files first, then fall back to text files
        try:
            camera_file = os.path.join(dataset_path, dataset_folder, 'sparse/0/cameras.bin')
            image_file = os.path.join(dataset_path, dataset_folder, 'sparse/0/images.bin')
            points3D_file = os.path.join(dataset_path, dataset_folder,'sparse/0/points3D.bin')
            cameras = read_intrinsics_binary(camera_file)
            images = read_extrinsics_binary(image_file)
            points3D = read_points3D_binary(points3D_file)
        except:
            camera_file = os.path.join(dataset_path, dataset_folder, 'colmap/cameras_undistorted.txt')
            image_file = os.path.join(dataset_path, dataset_folder, 'colmap/images.txt')
            points3D_file = os.path.join(dataset_path, dataset_folder,'colmap/points3D.txt')            
            cameras = read_intrinsics_text(camera_file)
            images = read_extrinsics_text(image_file)
            points3D = read_points3D_text(points3D_file)

        converter = ID2RGBConverter()

        # Apply selected voting algorithm
        if args.algorithm == 'majority':
            print("Using majority voting...")
            majority_voting(images, points3D, cameras, label_image_dir, color_image_dir, converter, output_ply_path)
        elif args.algorithm == 'prob':
            print("Using probability-based voting...")
            prob_voting(images, points3D, cameras, label_image_dir, color_image_dir, converter, output_ply_path)
        elif args.algorithm == 'corr':
            print("Using correlation-based voting...")
            corr_voting(images, points3D, label_image_dir, converter, output_ply_path)
        else:
            raise ValueError("Unknown algorithm. Choose from 'majority', 'prob', or 'corr'.")


if __name__ == "__main__":
    """Command line interface for point cloud preprocessing.
    
    Supports three voting algorithms:
    - majority: Simple majority voting
    - prob: Probability-based voting with random sampling
    - corr: Correlation-based voting using track correspondences
    """
    parser = argparse.ArgumentParser(
        description="Preprocess 3D point clouds with semantic labels using various voting strategies."
    )
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        default='datasets/lerf_mask', 
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '--algorithm', 
        type=str, 
        default='corr', 
        choices=['majority', 'prob', 'corr'], 
        help='Voting algorithm to use'
    )
    parser.add_argument(
        '--output_ply_name', 
        type=str, 
        default='points3D_corr.ply', 
        help='Output PLY file name'
    )
    args = parser.parse_args()
    main(args)