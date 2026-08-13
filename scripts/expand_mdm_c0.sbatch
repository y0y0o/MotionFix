#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=02:00:00
#SBATCH --job-name=mfix_mdm_c0
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/expand_mdm_c0.log

source /etc/profile
source ~/miniconda3/bin/activate mdm
module purge
module load cuda/11.0-cudnn8.0.5

MODEL=/home3/nxkh91/projects/mdm/save/models_to_upload/humanml_trans_dec_512_bert/model000200000.pt
EXP=/home3/nxkh91/projects/motionfix/data/test_inputs/_expand
DIR=$(dirname $MODEL)
cd /home3/nxkh91/projects/mdm
echo "START $(date) on $(hostname)"

# warmup 进程:吸收新分配GPU上首个CUDA进程的瞬时失败(原作业里正是第一个进程整批NaN)
echo "=== warmup ==="
python -m sample.generate --model_path $MODEL --input_text $EXP/mdm_warmup.txt \
  --num_repetitions 1 --seed 10 --device 0 || true

# 生成 chunk0,自检NaN,最多重试3次(每次换seed避开同一坏抽样)
R0=$DIR/samples_humanml_trans_dec_512_bert_000200000_seed*_mdm_text_c0/results.npy
for try in 1 2 3; do
  SEED=$((10 + try - 1))   # 10,11,12
  echo "=== chunk0 try $try seed $SEED ==="
  rm -rf $DIR/samples_humanml_trans_dec_512_bert_000200000_seed${SEED}_mdm_text_c0
  python -m sample.generate --model_path $MODEL --input_text $EXP/mdm_text_c0.txt \
    --num_repetitions 1 --seed $SEED --device 0 || true
  RES=$DIR/samples_humanml_trans_dec_512_bert_000200000_seed${SEED}_mdm_text_c0/results.npy
  # NaN 自检
  python -c "
import numpy as np, sys
d=np.load('$RES',allow_pickle=True).item(); mot=d['motion']; lens=d['lengths']
bad=sum(1 for i in range(mot.shape[0]) if not np.isfinite(mot[i].transpose(2,0,1)[:int(lens[i])]).all())
print('  chunk0 含NaN条数:',bad,'/',mot.shape[0])
sys.exit(1 if bad>0 else 0)
"
  if [ $? -eq 0 ]; then
    echo "=== chunk0 clean, convert ==="
    python $EXP/mdm_convert.py $RES $EXP/mdm_pids_c0.txt
    echo "END $(date)"
    exit 0
  fi
  echo "  try $try 仍有NaN,重试"
done
echo "FAILED chunk0 after 3 tries $(date)"
exit 1
