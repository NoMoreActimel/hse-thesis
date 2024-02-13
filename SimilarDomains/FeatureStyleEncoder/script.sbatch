#!/bin/bash
#SBATCH --job-name=feature-style-encoder-inference
#SBATCH --error=runs/feature-style-encoder-inference-%j.err
#SBATCH --output=runs/feature-style-encoder-inference-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:15:00
#SBATCH --constraint="[type_a|type_b|type_c]"

module load Python/Anaconda_v03.2023

source deactivate
source activate FSE_env

python3 test.py --pretrained_model_path './pretrained_models/143_enc.pth' --input-path './test/'
