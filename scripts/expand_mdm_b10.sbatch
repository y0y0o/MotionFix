#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=04:00:00
#SBATCH --job-name=mfix_mdm_b10
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/expand_mdm_b10.log
source /etc/profile
source ~/miniconda3/bin/activate mdm
module purge; module load cuda/11.0-cudnn8.0.5
MODEL=/home3/nxkh91/projects/mdm/save/models_to_upload/humanml_trans_dec_512_bert/model000200000.pt
DIR=$(dirname $MODEL)
EXP=/home3/nxkh91/projects/motionfix/data/test_inputs/_expand
cd /home3/nxkh91/projects/mdm
echo "START $(date) on $(hostname)"

RESARGS=""
for k in $(seq 0 14); do
  echo "=== batch $k (10 prompts, 200k+ema) ==="
  rm -rf $DIR/samples_humanml_trans_dec_512_bert_000200000_seed10_text_${k}
  python -m sample.generate --model_path $MODEL --input_text $EXP/b10/text_${k}.txt \
    --num_repetitions 1 --seed 10 --device 0 --use_ema
  echo "  >>> batch$k exit=$?"
  RES=$DIR/samples_humanml_trans_dec_512_bert_000200000_seed10_text_${k}/results.npy
  [ -f "$RES" ] && RESARGS="$RESARGS $RES $EXP/b10/pids_${k}.txt"
done

echo "=== convert ==="
python $EXP/mdm_convert.py $RESARGS
echo "END $(date)"
