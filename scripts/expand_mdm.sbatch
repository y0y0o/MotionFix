#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=06:00:00
#SBATCH --job-name=mfix_mdm_gen
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/expand_mdm.log

source /etc/profile
source ~/miniconda3/bin/activate mdm
module purge
module load cuda/11.0-cudnn8.0.5

MODEL=/home3/nxkh91/projects/mdm/save/models_to_upload/humanml_trans_dec_512_bert/model000200000.pt
EXP=/home3/nxkh91/projects/motionfix/data/test_inputs/_expand
cd /home3/nxkh91/projects/mdm
echo "START $(date) on $(hostname)"

# 分3块生成(每块50,避免12GB显存OOM),seed10 匹配现有50条
RESARGS=""
for c in 0 1 2; do
  echo "=== chunk $c ==="
  python -m sample.generate \
    --model_path $MODEL \
    --input_text $EXP/mdm_text_c${c}.txt \
    --num_repetitions 1 \
    --seed 10 \
    --device 0
  RES=$(dirname $MODEL)/samples_humanml_trans_dec_512_bert_000200000_seed10_mdm_text_c${c}/results.npy
  RESARGS="$RESARGS $RES $EXP/mdm_pids_c${c}.txt"
done

echo "=== convert ==="
python $EXP/mdm_convert.py $RESARGS
echo "END $(date)"
