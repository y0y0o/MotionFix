"""
把 MDM 各 chunk 的 results.npy 转成 mdm_{pid}_joints.npy 存入 mdm_raw_joints/。
results['motion']: (N,22,3,T)  -> transpose (T,22,3)，按 chunk 内顺序映射到 pid。
用法: python mdm_convert.py <results_c0.npy> mdm_pids_c0.txt  [更多对...]
"""
import sys, os, numpy as np

OUT = "/home3/nxkh91/projects/motionfix/data/test_inputs/mdm/mdm_raw_joints"
os.makedirs(OUT, exist_ok=True)

args = sys.argv[1:]
pairs = [(args[i], args[i+1]) for i in range(0, len(args), 2)]
total = 0
for res_path, pid_path in pairs:
    if not os.path.exists(res_path):
        print(f"MISSING {res_path}"); continue
    data = np.load(res_path, allow_pickle=True).item()
    motions = data['motion']            # (N,22,3,T)
    lengths = data['lengths']
    pids = [l.strip() for l in open(pid_path) if l.strip()]
    N = motions.shape[0]
    if N != len(pids):
        print(f"WARN {res_path}: N={N} != pids={len(pids)} — 按 min 映射")
    for i in range(min(N, len(pids))):
        L = int(lengths[i])
        m = motions[i].transpose(2, 0, 1)[:L]   # (T,22,3)
        np.save(f"{OUT}/mdm_{pids[i]}_joints.npy", m.astype(np.float32))
        total += 1
    print(f"{res_path}: 转出 {min(N,len(pids))} 条")
print(f"DONE 共 {total} 条 -> {OUT}")
