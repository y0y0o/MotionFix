#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=03:00:00
#SBATCH --job-name=mfix_mdm_c0sub
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/expand_mdm_c0sub.log

source /etc/profile
source ~/miniconda3/bin/activate mdm
module purge
module load cuda/11.0-cudnn8.0.5

MODEL=/home3/nxkh91/projects/mdm/save/models_to_upload/humanml_trans_dec_512_bert/model000200000.pt
EXP=/home3/nxkh91/projects/motionfix/data/test_inputs/_expand
DIR=$(dirname $MODEL)
cd /home3/nxkh91/projects/mdm
echo "START $(date) on $(hostname)"

clean=0; dirty=""
for k in 0 1 2 3 4 5 6 7 8 9; do
  echo "=== sub $k ==="
  rm -rf $DIR/samples_humanml_trans_dec_512_bert_000200000_seed10_text_${k}
  python -m sample.generate --model_path $MODEL --input_text $EXP/c0sub/text_${k}.txt \
    --num_repetitions 1 --seed 10 --device 0 || true
  RES=$DIR/samples_humanml_trans_dec_512_bert_000200000_seed10_text_${k}/results.npy
  python -c "
import numpy as np, sys
d=np.load('$RES',allow_pickle=True).item(); mot=d['motion']; lens=d['lengths']
bad=sum(1 for i in range(mot.shape[0]) if not np.isfinite(mot[i].transpose(2,0,1)[:int(lens[i])]).all())
print('  sub$k NaN:',bad,'/',mot.shape[0]); sys.exit(1 if bad>0 else 0)
"
  if [ $? -eq 0 ]; then
    python $EXP/mdm_convert.py $RES $EXP/c0sub/pids_${k}.txt
    clean=$((clean+1))
  else
    dirty="$dirty $k"
  fi
done

echo "=== SUMMARY: clean_subs=$clean/10  dirty_subs=[$dirty] ==="
echo "END $(date)"
