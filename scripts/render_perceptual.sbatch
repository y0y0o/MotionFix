#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=02:00:00
#SBATCH --job-name=mfix_render_perc
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/render_perceptual.log
source /etc/profile
source ~/miniconda3/bin/activate t2mgpt
export MPLBACKEND=Agg
cd /home3/nxkh91/projects/motionfix
echo "START $(date)"
python utils/render_perceptual.py
echo "clips: $(ls outputs/perceptual/clips/*.mp4 2>/dev/null | wc -l)"
echo "END $(date)"
