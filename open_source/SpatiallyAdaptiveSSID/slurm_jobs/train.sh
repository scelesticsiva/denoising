#!/usr/bin/bash

#SBATCH --job-name=ssid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_geforce_gtx_1080_ti:1
#SBATCH --exclusive
#SBATCH --mem=32g
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate mibi_segmentation
python3 /home/schaudhary/siva_projects/SpatiallyAdaptiveSSID/train.py\
    --config /home/schaudhary/siva_projects/SpatiallyAdaptiveSSID/option/$1