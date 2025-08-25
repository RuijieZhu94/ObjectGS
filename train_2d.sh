#!/bin/bash

# Set the path to the scene directory
SCENE_DIR=$1
scene_dataset=$(basename "$SCENE_DIR")

# Iterate through all scene folders in the specified directory
for scene in "$SCENE_DIR"/*; do
  # Ensure only directories execute the training command
  if [ -d "$scene" ]; then
    # Execute the training command
    echo "Training scene: $scene"
    scene_name=$(basename "$scene")
    python train.py --config config/objectgs/2d/$scene_dataset/config.yaml --scene_name $scene_name
  fi
done