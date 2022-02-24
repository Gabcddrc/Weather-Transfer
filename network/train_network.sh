#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=sichen.li@psi.ch
#SBATCH --clusters=gmerlin6
#SBATCH --partition=gwendolen
#SBATCH --account=gwendolen
#SBATCH --partition=gwendolen
#SBATCH --gpus=4
#SBATCH --mem=800000
# SBATCH --array=5
#SBATCH --time=1:00:00
#SBATCH -o train_network.out # logfile for STDOUT
#SBATCH -e train_network.err # logfile for STDERR

python /psi/home/li_s1/DeepLearning2021/network/train.py
