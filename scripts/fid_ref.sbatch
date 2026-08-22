#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=02:00:00
#SBATCH --job-name=mfix_fid_ref
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/fid_ref.log
source /etc/profile
source ~/miniconda3/bin/activate t2mgpt
cd /home3/nxkh91/projects/motionfix
echo "START $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python testing/v19_fid_ref.py --v19 checkpoints/v19_088a10/best.pth
echo "END $(date)"
