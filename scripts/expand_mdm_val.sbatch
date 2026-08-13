#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --time=00:40:00
#SBATCH --job-name=mfix_mdm_val
#SBATCH --output=/home3/nxkh91/projects/motionfix/logs/expand_mdm_val.log
source /etc/profile
source ~/miniconda3/bin/activate mdm
module purge; module load cuda/11.0-cudnn8.0.5
MODEL=/home3/nxkh91/projects/mdm/save/models_to_upload/humanml_trans_dec_512_bert/model000600000.pt
EXP=/home3/nxkh91/projects/motionfix/data/test_inputs/_expand
cd /home3/nxkh91/projects/mdm
echo "START $(date)"
python -m sample.generate --model_path $MODEL --input_text $EXP/mdm_val10.txt \
  --num_repetitions 1 --seed 10 --device 0 --use_ema || true
RES=$(dirname $MODEL)/samples_humanml_trans_dec_512_bert_000600000_seed10_mdm_val10/results.npy
python -c "
import numpy as np
d=np.load('$RES',allow_pickle=True).item(); mot=d['motion']; lens=d['lengths']
fvs=[]
for i in range(mot.shape[0]):
    L=int(lens[i]); m=mot[i].transpose(2,0,1)[:L]
    fv=np.linalg.norm(np.diff(m[:,[7,8,10,11]][:,:,[0,2]],axis=0),axis=-1)
    fvs.append(fv.mean())
import numpy as np
print('600k+EMA footV_mean: min%.4f mean%.4f max%.4f (正常应~0.03,垃圾是~0.34)'%(min(fvs),float(np.mean(fvs)),max(fvs)))
finite=all(np.isfinite(mot[i]).all() for i in range(mot.shape[0]))
print('all finite:',finite)
"
echo "END $(date)"
