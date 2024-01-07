#!/bin/bash
#SBATCH -J care_3D
#SBATCH -A gts-hl94-joe
#SBATCH -q inferno
#SBATCH -t 24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH -o %j.out

module load anaconda3/2022.05
conda activate baseline_denoising
module load cuda
python /storage/home/hcoda1/0/schaudhary9/p-hl94-0/siva_projects/baseline_denoising/train_care_3D.py
conda deactivate