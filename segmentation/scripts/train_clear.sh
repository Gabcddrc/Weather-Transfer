#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=sichen.li@psi.ch
#SBATCH --clusters=gmerlin6
#SBATCH --account=gwendolen
#SBATCH --partition=gwendolen
#SBATCH --gpus=4
#SBATCH --mem=100000
#SBATCH --array=0-5
#SBATCH --time=1:00:00
#SBATCH -o train_clear_%a.out # logfile for STDOUT
#SBATCH -e train_clear_%a.err # logfile for STDERR

python /psi/home/li_s1/DeepLearning2021/segmentation/run_segmentation.py train clear $SLURM_ARRAY_TASK_ID
