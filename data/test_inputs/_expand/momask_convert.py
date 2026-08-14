"""
把 MoMask gen_t2m 的 no_ik 输出映射到 pid,存进 momask_new/ 暂存目录。
gen 输出: generation/<ext>/joints/<k>/sample<k>_repeat0_len<L>.npy  (no_ik, 无 _ik 后缀)
按 chunk 内 k 顺序映射到 pids 文件。
用法: python momask_convert.py <gen_root> <pids_file>  [更多对...]
"""
import sys, os, glob, numpy as np

OUT = "/home3/nxkh91/projects/motionfix/data/test_inputs/_expand/momask_new"
os.makedirs(OUT, exist_ok=True)
args = sys.argv[1:]
pairs = [(args[i], args[i+1]) for i in range(0, len(args), 2)]
total = 0
for gen_root, pid_path in pairs:
    pids = [l.strip() for l in open(pid_path) if l.strip()]
    jdir = os.path.join(gen_root, 'joints')
    for k, pid in enumerate(pids):
        cand = [f for f in glob.glob(f"{jdir}/{k}/sample{k}_repeat0_len*.npy") if not f.endswith('_ik.npy')]
        if not cand:
            print(f"  MISSING k={k} pid={pid}"); continue
        a = np.load(cand[0]).astype(np.float32)
        np.save(f"{OUT}/p{pid}_expand.npy", a)
        total += 1
    print(f"{gen_root}: 映射 {len(pids)} 条")
print(f"DONE 共 {total} 条 -> {OUT}")
