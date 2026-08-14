"""
Pooled MoMask eval (n=200: 50 原始 + 150 扩充).
复用 v19_eval 的方法与指标,对 momask_pool/ 做五路消融,避开 v19_eval 的 heldout/train 特殊拆分。
输出: analysis/v19/momask_pool_results.json
用法: python testing/momask_pool_eval.py --v19 checkpoints/v19_088a10/best.pth
"""
import torch, numpy as np, os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v18 import FootRefiner, smooth_fix
from models.v18_ik import apply_ik
from models.v19 import V19Smoother, smooth_fix_v19
from testing.v19_eval import M, deskate_only, avg, load, GAUSS_SIGMA, V18_CKPT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
POOL = "data/test_inputs/momask_pool"
OUT = "analysis/v19/momask_pool_results.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v19', default='checkpoints/v19_088a10/best.pth')
    args = ap.parse_args()

    m18 = FootRefiner().to(DEVICE)
    m18.load_state_dict(torch.load(V18_CKPT, map_location=DEVICE)['model_state_dict']); m18.eval()
    m19 = V19Smoother().to(DEVICE)
    m19.load_state_dict(torch.load(args.v19, map_location=DEVICE)['model_state_dict']); m19.eval()

    files = sorted(glob.glob(f"{POOL}/*.npy"))
    print(f"pooled momask: {len(files)} motions")
    methods = ['original', 'deskate_ik', 'gauss_ik', 'learn_ik', 'v19_ik']
    acc = {k: [] for k in methods}
    for fp in files:
        mo = load(fp)
        acc['original'].append(M(mo, mo))
        acc['deskate_ik'].append(M(apply_ik(mo, deskate_only(mo, 0.0)), mo))
        acc['gauss_ik'].append(M(apply_ik(mo, deskate_only(mo, GAUSS_SIGMA)), mo))
        acc['learn_ik'].append(M(apply_ik(mo, smooth_fix(mo, m18, DEVICE)), mo))
        acc['v19_ik'].append(M(apply_ik(mo, smooth_fix_v19(mo, m19, DEVICE)), mo))
    res = {k: avg(v) for k, v in acc.items()}
    res['n'] = len(files)

    print(f"\n{'method':<12} {'FSR':>7} {'Jitter':>9} {'FootErr':>8}")
    for k in methods:
        print(f"  {k:<12} {res[k]['FSR']*100:6.2f}% {res[k]['Jitter']:9.5f} {res[k]['FootErr']:8.4f}")
    d = res['deskate_ik']['FSR']*100
    print(f"  平滑相对deskate: gauss {res['gauss_ik']['FSR']*100-d:+.2f} | learn {res['learn_ik']['FSR']*100-d:+.2f} | v19 {res['v19_ik']['FSR']*100-d:+.2f}")

    json.dump({'momask_pool': res}, open(OUT, 'w'), indent=2)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
