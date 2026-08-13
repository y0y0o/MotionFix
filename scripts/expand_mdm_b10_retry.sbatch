#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=05:00:00
#SBATCH --job-name=mfix_mdm_retry
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/expand_mdm_b10_retry.log
source /etc/profile
source ~/miniconda3/bin/activate mdm
module purge; module load cuda/11.0-cudnn8.0.5
MODEL=/home3/nxkh91/projects/mdm/save/models_to_upload/humanml_trans_dec_512_bert/model000200000.pt
DIR=$(dirname $MODEL)
EXP=/home3/nxkh91/projects/motionfix/data/test_inputs/_expand
cd /home3/nxkh91/projects/mdm
echo "START $(date) on $(hostname)"

MISSING="0 3 4 5 6 8 9 10 11 12"
RESARGS=""
for k in $MISSING; do
  RES=$DIR/samples_humanml_trans_dec_512_bert_000200000_seed10_text_${k}/results.npy
  for try in 1 2 3; do
    [ -f "$RES" ] && break
    echo "=== batch $k try $try ==="
    rm -rf $DIR/samples_humanml_trans_dec_512_bert_000200000_seed10_text_${k}
    python -m sample.generate --model_path $MODEL --input_text $EXP/b10/text_${k}.txt \
      --num_repetitions 1 --seed 10 --device 0 --use_ema
    echo "  >>> batch$k try$try exit=$?"
  done
  [ -f "$RES" ] && RESARGS="$RESARGS $RES $EXP/b10/pids_${k}.txt" && echo "  batch$k OK" || echo "  batch$k 3次仍失败"
done

echo "=== convert ==="
python $EXP/mdm_convert.py $RESARGS
echo "END $(date)"
