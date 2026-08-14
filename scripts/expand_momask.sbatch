#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=03:00:00
#SBATCH --job-name=mfix_momask_gen
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/expand_momask.log
source /etc/profile
source ~/miniconda3/bin/activate t2mgpt
export MPLBACKEND=Agg
EXP=/home3/nxkh91/projects/motionfix/data/test_inputs/_expand
cd /home3/nxkh91/projects/momask-codes
echo "START $(date) on $(hostname)"

CONVARGS=""
for c in 0 1 2; do
  echo "=== chunk $c (50 prompts) ==="
  rm -rf generation/expand_c${c}
  python gen_t2m.py --gpu_id 0 --ext expand_c${c} \
    --text_path $EXP/mdm_text_c${c}.txt --motion_length 0 --repeat_times 1
  echo "  >>> chunk$c exit=$?"
  CONVARGS="$CONVARGS /home3/nxkh91/projects/momask-codes/generation/expand_c${c} $EXP/mdm_pids_c${c}.txt"
done

echo "=== convert ==="
python $EXP/momask_convert.py $CONVARGS
echo "END $(date)"
