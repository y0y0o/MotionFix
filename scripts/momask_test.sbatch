#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=00:40:00
#SBATCH --job-name=mfix_momask_test
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/momask_test.log
source /etc/profile
source ~/miniconda3/bin/activate t2mgpt
# no cuda module: t2mgpt torch bundles cu121
export MPLBACKEND=Agg
cd /home3/nxkh91/projects/momask-codes
echo "START $(date) on $(hostname)  torch=$(python -c 'import torch;print(torch.__version__)')"
python gen_t2m.py --gpu_id 0 --ext test2 --text_path test2_prompts.txt --motion_length 0 --repeat_times 1
echo ">>> gen exit=$?"
echo "产出:"; find generation/test2 -name "*.npy" 2>/dev/null | head
echo "END $(date)"
