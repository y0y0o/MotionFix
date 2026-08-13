#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=02:00:00
#SBATCH --job-name=mfix_eval_exp
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/expand_eval.log

source /etc/profile
source ~/miniconda3/bin/activate t2mgpt
cd /home3/nxkh91/projects/motionfix
echo "START $(date) on $(hostname)"

# 扩样本后重跑五路消融 7 指标(momask 50 / mdm 165 / t2mgpt 200)
python testing/v19_eval.py --v19 checkpoints/v19_088a10/best.pth --tag v19_088a10_expanded

echo "END $(date)"
