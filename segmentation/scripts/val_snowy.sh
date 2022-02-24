#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=sichen.li@psi.ch
#SBATCH --clusters=gmerlin6
#SBATCH --account=gwendolen
#SBATCH --partition=gwendolen
#SBATCH --gpus=2
#SBATCH --mem=100000
#SBATCH --array=0-1
#SBATCH --time=0:30:00
#SBATCH -o val_snowy_%a.out # logfile for STDOUT
#SBATCH -e val_snowy_%a.err # logfile for STDERR

python /psi/home/li_s1/DeepLearning2021/segmentation/run_segmentation.py val snowy $SLURM_ARRAY_TASK_ID 2
