#!/bin/bash

# Define the experiment directory path
experiment_dir=$1

# Get all scenes (by subfolder names)
scenes=$(ls -d ${experiment_dir}/*)

# Iterate through each scene
for scene_path in ${scenes}; do
    # Extract the scene name (remove the prefix part of the path)
    scene_name=$(basename ${scene_path})

    # Find the latest directory in the scene (sorted by date, select the latest)
    latest_dir=$(ls -dt ${scene_path}/* | head -n 1)

    # Execute the render.py script
    echo "Running render.py for scene: ${scene_name} with latest directory: ${latest_dir}"
    python render.py -m ${latest_dir} --scene_name ${scene_name} --skip_train
    # python render.py -m ${latest_dir} --scene_name ${scene_name} --skip_test
    python metrics.py -m ${latest_dir}
done