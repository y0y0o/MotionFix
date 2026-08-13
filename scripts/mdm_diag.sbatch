#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=00:50:00
#SBATCH --job-name=mfix_mdm_diag
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/mdm_diag.log
source /etc/profile
source ~/miniconda3/bin/activate mdm
module purge; module load cuda/11.0-cudnn8.0.5
DIR=/home3/nxkh91/projects/mdm/save/models_to_upload/humanml_trans_dec_512_bert
EXP=/home3/nxkh91/projects/motionfix/data/test_inputs/_expand
cd /home3/nxkh91/projects/mdm
echo "START $(date) on $(hostname)"

run_cfg () {
  local NITER=$1; local TAG=$2
  echo "########## CONFIG: model0${NITER} + use_ema  (tag=$TAG) ##########"
  rm -rf $DIR/samples_humanml_trans_dec_512_bert_0${NITER}_seed10_mdm_val10
  # 不吞错误,直接让 stderr 进日志
  python -m sample.generate --model_path $DIR/model0${NITER}.pt \
    --input_text $EXP/mdm_val10.txt --num_repetitions 1 --seed 10 --device 0 --use_ema
  echo "  >>> generate exit=$?"
  RES=$DIR/samples_humanml_trans_dec_512_bert_0${NITER}_seed10_mdm_val10/results.npy
  if [ -f "$RES" ]; then
    python -c "
import numpy as np
d=np.load('$RES',allow_pickle=True).item(); mot=d['motion']; lens=d['lengths']
import numpy as np
fvs=[np.linalg.norm(np.diff(mot[i].transpose(2,0,1)[:int(lens[i])][:,[7,8,10,11]][:,:,[0,2]],axis=0),axis=-1).mean() for i in range(mot.shape[0])]
print('  [$TAG] footV_mean mean=%.4f max=%.4f  finite=%s (正常~0.03,垃圾~0.34)'%(float(np.mean(fvs)),max(fvs),all(np.isfinite(mot[i]).all() for i in range(mot.shape[0]))))
"
  else
    echo "  [$TAG] NO results.npy — generation failed"
  fi
}

run_cfg 00200000 "200k+ema"
run_cfg 00600000 "600k+ema"
echo "END $(date)"
