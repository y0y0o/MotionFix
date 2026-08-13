#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=02:00:00
#SBATCH --job-name=mfix_t2mgpt_gen
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/expand_t2mgpt.log

source /etc/profile
source ~/miniconda3/bin/activate t2mgpt

cd /home3/nxkh91/projects/T2M-GPT
echo "START $(date) on $(hostname)"
python generate_t2mgpt_expand.py
echo "END $(date)"
