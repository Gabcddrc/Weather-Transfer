#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=sichen.li@psi.ch
#SBATCH --clusters=gmerlin6
#SBATCH --account=gwendolen
#SBATCH --partition=gwendolen
#SBATCH --gpus=4
#SBATCH --mem=800000
# SBATCH --array=0-5
#SBATCH --time=0:30:00
#SBATCH -o val_seg_retrain.out # logfile for STDOUT
#SBATCH -e val_seg_retrain.err # logfile for STDERR

python /psi/home/li_s1/DeepLearning2021/segmentation/main.py masks
