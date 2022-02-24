#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=sichen.li@psi.ch
#SBATCH --clusters=gmerlin6
#SBATCH --account=gwendolen
#SBATCH --partition=gwendolen_long
#SBATCH --gpus=4
#SBATCH --mem=800000
# SBATCH --array=0-5
#SBATCH --time=6:00:00
#SBATCH -o train_seg.out # logfile for STDOUT
#SBATCH -e train_seg.err # logfile for STDERR

python /psi/home/li_s1/DeepLearning2021/segmentation/main.py colormaps
