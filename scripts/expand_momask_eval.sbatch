#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=01:30:00
#SBATCH --job-name=mfix_momask_eval
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/expand_momask_eval.log
source /etc/profile
source ~/miniconda3/bin/activate t2mgpt
cd /home3/nxkh91/projects/motionfix
echo "START $(date)"
python testing/momask_pool_eval.py --v19 checkpoints/v19_088a10/best.pth
echo "END $(date)"
