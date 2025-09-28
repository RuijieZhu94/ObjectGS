# Point Cloud Preprocessing for ObjectGS

This module provides functionality for preprocessing 3D point clouds with semantic labels by projecting them onto 2D images and assigning colors/labels through various voting strategies.

## Features

- **Multiple Voting Algorithms**: Support for majority voting, probability-based voting, and correlation-based voting
- **COLMAP Integration**: Compatible with both binary and text COLMAP formats
- **Flexible Input**: Supports both color and label images
- **PLY Output**: Generates standard PLY files with semantic labels

## Usage

```bash
python ply_preprocessing.py --dataset_path /path/to/dataset --algorithm corr --output_ply_name points3D_corr.ply
```

### Command Line Arguments

- `--dataset_path`: Path to the dataset directory (default: `datasets/lerf_mask`)
- `--algorithm`: Voting algorithm to use. Options: `majority`, `prob`, `corr` (default: `corr`)
- `--output_ply_name`: Output PLY file name (default: `points3D_corr.ply`)

## Voting Algorithms

### 1. Majority Voting (`majority`)
- Assigns the most frequently occurring label to each 3D point
- Simple and robust approach
- Best for datasets with consistent labeling

### 2. Probability-based Voting (`prob`)
- Samples labels based on their probability distribution
- Introduces controlled randomness
- Useful for handling ambiguous regions

### 3. Correlation-based Voting (`corr`)
- Uses COLMAP track correspondences for more accurate assignment
- Leverages geometric consistency
- Recommended for high-accuracy requirements

## Directory Structure

The script expects the following directory structure for each dataset:

```
dataset_folder/
├── object_mask/          # Label images (grayscale)
├── color_mask/           # Optional: Color images (RGB)
└── sparse/0/             # COLMAP sparse reconstruction
    ├── cameras.bin       # Camera parameters
    ├── images.bin        # Image poses
    └── points3D.bin      # 3D points
```

Alternative COLMAP format:
```
dataset_folder/
├── object_mask/
├── color_mask/
└── colmap/
    ├── cameras_undistorted.txt
    ├── images.txt
    └── points3D.txt
```

## Output Format

The generated PLY file contains the following attributes for each point:
- `x, y, z`: 3D coordinates
- `nx, ny, nz`: Normal vectors (set to zero)
- `red, green, blue`: RGB color values
- `label`: Semantic label ID

## Key Classes and Functions

### `ID2RGBConverter`
Converts object IDs to unique RGB colors for visualization.

### Core Functions
- `majority_voting()`: Implements majority voting strategy
- `prob_voting()`: Implements probability-based voting strategy  
- `corr_voting()`: Implements correlation-based voting strategy
- `storePly()`: Saves point cloud to PLY format

### Utility Functions
- `project_points()`: Projects 3D points to 2D image plane
- `get_point_colors_from_image()`: Extracts colors and labels from images
- `quaternion_to_rotation_matrix()`: Converts quaternions to rotation matrices

## Requirements

- OpenCV (`cv2`)
- NumPy
- PLY file handling (`plyfile`)
- COLMAP loader utilities

## License

MIT License - See the main repository for details.