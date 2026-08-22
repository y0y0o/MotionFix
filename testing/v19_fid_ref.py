"""
Large-reference FID / semantic evaluation
=========================================
Fixes the one weakness the reviewer flagged in `testing/v19_semantic.py`:
the FID ground-truth reference was only n=26 (of the 50 eval prompts, just 26
had a ground-truth motion in the local HumanML3D checkout). A 26-sample
reference cannot fill a ~512-d embedding covariance, so FID was effectively the
mean term alone — noise.

What changed vs v19_semantic.py
-------------------------------
1. FID REFERENCE is now decoupled from the eval prompts. It is built from the
   *standard HumanML3D test split* (`test.txt`), using every id whose
   ground-truth `new_joint_vecs` file exists locally (~1215 motions) — the same
   set the published protocol measures FID against. Full-rank covariance.
2. GENERATED side is sample-driven, not prompt-driven, so it uses ALL available
   samples per generator (t2mgpt ~200, momask ~200, mdm ~50) instead of only the
   50 that matched `test_prompts_50.txt`.
3. Motion embedding is chunked so the 1215-motion reference fits in 12 GB.

Everything else (conversion path, evaluator, paired bootstrap) is unchanged, so
the numbers stay directly comparable to `analysis/v19/semantic.json`.

Usage:
    python testing/v19_fid_ref.py --v19 checkpoints/v19_088a10/best.pth
    python testing/v19_fid_ref.py --n-ref 800 --n-gen 200      # caps for a quick run
Output:
    analysis/v19/semantic_largeref.json,  logs/v19_fid_ref.log
"""
import os, sys, re, json, glob, argparse
from datetime import datetime

import numpy as np
import torch

_MF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_T2M = "/home3/nxkh91/projects/T2M-GPT"
_HML = "/home3/nxkh91/projects/HumanML3D"

sys.path.insert(0, _MF)
from utils.joints_to_feats import joints_to_263                              # noqa: E402
from models.v18 import FootRefiner, smooth_fix, deskate_xz, FOOT_XZ_DIMS     # noqa: E402
from models.v18_ik import apply_ik                                           # noqa: E402
from models.v19 import V19Smoother, smooth_fix_v19                           # noqa: E402
from scipy.ndimage import gaussian_filter1d                                  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# sample dirs + a regex that pulls the 6-digit HumanML3D id out of the filename
GENERATORS = {
    'momask':  ("data/test_inputs/momask_pool",                 r"p(\d{6})"),
    'mdm':     ("data/test_inputs/mdm/mdm_raw_joints",          r"mdm_(\d{6})"),
    't2mgpt':  ("data/test_inputs/t2mgpt/t2mgpt_raw_joints",    r"t2mgpt_(\d{6})"),
}
TEST_SPLIT = f"{_HML}/HumanML3D/test.txt"
TEXTS = f"{_HML}/HumanML3D/texts"
GT_VECS = f"{_HML}/HumanML3D/new_joint_vecs"
V18_CKPT = "checkpoints/v18_ik/best.pth"
GAUSS_SIGMA = 1.5
ANA, LOG = "analysis/v19", "logs/v19_fid_ref.log"
MAX_LEN, UNIT = 196, 4
N_RPREC_POOL, N_RPREC_REPEAT = 32, 200
EMBED_CHUNK = 64
os.makedirs(ANA, exist_ok=True); os.makedirs(os.path.dirname(LOG), exist_ok=True)


def log(m):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}"
    with open(LOG, 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


# ══════════════════════════════════════════════════════════════════════════
# Phase 1 — build features (263-d) for the reference and for every generator
# ══════════════════════════════════════════════════════════════════════════

def load_joints(fp):
    m = np.load(fp).astype(np.float32)
    return m[0] if m.ndim == 4 else m


def deskate_only(m, sigma=0.0):
    T = m.shape[0]
    flat = m.reshape(T, -1).astype(np.float32)
    tgt, _ = deskate_xz(m)
    if sigma > 0:
        tgt = gaussian_filter1d(tgt, sigma=sigma, axis=0, mode='nearest')
    out = flat.copy(); out[:, FOOT_XZ_DIMS] = tgt
    return out.reshape(T, 22, 3)


def first_tokens(mid):
    """POS tokens from the first annotation of a motion, or None if missing."""
    tf = f"{TEXTS}/{mid}.txt"
    if not os.path.exists(tf):
        return None
    parts = open(tf).readline().strip().split('#')
    if len(parts) < 2:
        return None
    return parts[1].split(' ')


def _recover_from_ric():
    """Load T2M-GPT's recover_from_ric without letting its utils.* shadow ours."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('t2m_quaternion',
                                                  f"{_T2M}/utils/quaternion.py")
    q = importlib.util.module_from_spec(spec); sys.modules['t2m_quaternion'] = q
    spec.loader.exec_module(q)
    src = open(f"{_T2M}/utils/motion_process.py").read().replace(
        "from utils.quaternion import", "from t2m_quaternion import")
    mp = importlib.util.module_from_spec(
        importlib.util.spec_from_loader('t2m_motion_process', loader=None))
    exec(compile(src, 'motion_process.py', 'exec'), mp.__dict__)
    return mp.recover_from_ric


def build_reference(n_ref):
    """Large FID reference: local test-split GT motions -> (T,263), + tokens."""
    recover = _recover_from_ric()
    feats, toks = [], []
    for line in open(TEST_SPLIT):
        mid = line.strip()
        if not mid:
            continue
        fp = f"{GT_VECS}/{mid}.npy"
        tk = first_tokens(mid)
        if not os.path.exists(fp) or tk is None:
            continue
        j = recover(torch.from_numpy(np.load(fp)).float(), 22).numpy()
        feats.append(joints_to_263(j[:MAX_LEN])); toks.append(tk)
        if n_ref and len(feats) >= n_ref:
            break
    log(f"  FID reference: {len(feats)} ground-truth test-split motions")
    return feats, toks


def build_generated(args):
    """feats[gen][method] = list of (T,263); + parallel token list per gen."""
    m18 = FootRefiner().to(DEVICE)
    m18.load_state_dict(torch.load(V18_CKPT, map_location=DEVICE)['model_state_dict'])
    m18.eval()
    m19 = None
    if args.v19:
        m19 = V19Smoother().to(DEVICE)
        m19.load_state_dict(torch.load(args.v19, map_location=DEVICE)['model_state_dict'])
        m19.eval()
    methods = ['original', 'deskate_ik', 'gauss_ik', 'learn_ik'] + (['v19_ik'] if m19 else [])

    out = {}
    for gen, (d, pat) in GENERATORS.items():
        rgx = re.compile(pat)
        acc = {k: [] for k in methods}
        toks = []
        files = sorted(glob.glob(os.path.join(d, "*.npy")))
        seen = set()
        for fp in files:
            mm = rgx.search(os.path.basename(fp))
            if not mm:
                continue
            mid = mm.group(1)
            if mid in seen:                       # one sample per id
                continue
            tk = first_tokens(mid)
            if tk is None:
                continue
            mo = load_joints(fp)[:MAX_LEN]
            try:
                row = {
                    'original':   joints_to_263(mo),
                    'deskate_ik': joints_to_263(apply_ik(mo, deskate_only(mo, 0.0))),
                    'gauss_ik':   joints_to_263(apply_ik(mo, deskate_only(mo, GAUSS_SIGMA))),
                    'learn_ik':   joints_to_263(apply_ik(mo, smooth_fix(mo, m18, DEVICE))),
                }
                if m19:
                    row['v19_ik'] = joints_to_263(apply_ik(mo, smooth_fix_v19(mo, m19, DEVICE)))
            except Exception as e:                 # noqa: BLE001
                log(f"    !! {gen}/{mid} conversion failed ({e}) — dropped")
                continue
            for k in methods:
                acc[k].append(row[k])
            toks.append(tk); seen.add(mid)
            if args.n_gen and len(toks) >= args.n_gen:
                break
        out[gen] = acc
        out[gen + '__toks'] = toks
        log(f"  {gen}: {len(toks)} motions x {len(methods)} methods converted")
    return out, methods


# ══════════════════════════════════════════════════════════════════════════
# Phase 2 — evaluator (runs inside the T2M-GPT import context)
# ══════════════════════════════════════════════════════════════════════════

def evaluate(ref_feats, ref_toks, gen_feats, methods):
    cwd = os.getcwd()
    saved_mods = {k: v for k, v in sys.modules.items()
                  if k in ('utils', 'models', 'options') or
                  k.startswith(('utils.', 'models.', 'options.'))}
    for k in saved_mods:
        del sys.modules[k]
    saved_path = list(sys.path)
    sys.path = [p for p in sys.path if p not in ('', _MF)]
    sys.path.insert(0, _T2M)
    os.chdir(_T2M)

    try:
        from options.get_eval_option import get_opt
        from models.evaluator_wrapper import EvaluatorModelWrapper
        from utils.word_vectorizer import WordVectorizer
        from utils.eval_trans import (calculate_R_precision, calculate_diversity,
                                       calculate_activation_statistics,
                                       calculate_frechet_distance,
                                       euclidean_distance_matrix)

        wrapper_opt = get_opt('checkpoints/t2m/Comp_v6_KLD005/opt.txt', torch.device(DEVICE))
        ev = EvaluatorModelWrapper(wrapper_opt)
        wv = WordVectorizer('./glove', 'our_vab')
        mean = np.load('/home3/nxkh91/projects/mdm/dataset/t2m_mean.npy')
        std = np.load('/home3/nxkh91/projects/mdm/dataset/t2m_std.npy')

        def text_batch(tok_list):
            we, po, cl = [], [], []
            for tk in tok_list:
                tk = tk[:wrapper_opt.max_text_len]
                sent = ['sos/OTHER'] + tk + ['eos/OTHER']
                cl.append(len(sent))
                sent = sent + ['unk/OTHER'] * (wrapper_opt.max_text_len + 2 - len(sent))
                w, p = [], []
                for t in sent:
                    a, b = wv[t]
                    w.append(a[None, :]); p.append(b[None, :])
                we.append(np.concatenate(w, 0)[None])
                po.append(np.concatenate(p, 0)[None])
            return (torch.from_numpy(np.concatenate(we, 0)),
                    torch.from_numpy(np.concatenate(po, 0)),
                    torch.tensor(cl))

        def motion_batch(fl):
            ms, ml = [], []
            for f in fl:
                T = min(len(f), MAX_LEN) // UNIT * UNIT
                x = (f[:T] - mean) / std
                ml.append(T)
                ms.append(np.pad(x, ((0, MAX_LEN - T), (0, 0)))[None])
            return torch.from_numpy(np.concatenate(ms, 0)).float(), torch.tensor(ml)

        def embed(fl, tk):
            """Chunked co-embedding so a 1000+ reference fits in memory."""
            T_out, M_out = [], []
            for s in range(0, len(fl), EMBED_CHUNK):
                fchunk, tchunk = fl[s:s + EMBED_CHUNK], tk[s:s + EMBED_CHUNK]
                m, ml = motion_batch(fchunk)
                w, p, cl = text_batch(tchunk)
                order = torch.argsort(cl, descending=True)
                inv = torch.argsort(order)
                t_emb, m_emb = ev.get_co_embeddings(
                    w[order], p[order], cl[order], m[order], ml[order])
                T_out.append(t_emb.cpu().numpy()[inv.numpy()])
                M_out.append(m_emb.cpu().numpy()[inv.numpy()])
            return np.concatenate(T_out, 0), np.concatenate(M_out, 0)

        def rprec_and_dist(t_emb, m_emb):
            n = len(t_emb)
            rng = np.random.default_rng(0)
            tops = []
            for _ in range(N_RPREC_REPEAT):
                idx = rng.choice(n, min(N_RPREC_POOL, n), replace=False)
                d = euclidean_distance_matrix(t_emb[idx], m_emb[idx])
                order = np.argsort(d, axis=1)
                hit = (order == np.arange(len(idx))[:, None])
                tops.append([hit[:, :k].any(1).mean() for k in (1, 2, 3)])
            ps = np.linalg.norm(t_emb - m_emb, axis=1)
            return np.mean(tops, 0), float(ps.mean()), ps

        def paired_bootstrap(a, b, n_boot=10000, seed=0):
            d = a - b
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(d), (n_boot, len(d)))
            boot = d[idx].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            return float(d.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)

        # ── large, full-rank ground-truth reference ──
        _, ref_m = embed(ref_feats, ref_toks)
        gt_mu, gt_cov = calculate_activation_statistics(ref_m)
        log(f"  reference statistics from {len(ref_m)} motions "
            f"(embedding dim {ref_m.shape[1]})")

        out = {'_fid_reference_n': len(ref_m),
               '_embedding_dim': int(ref_m.shape[1])}
        for gen in GENERATORS:
            tk = gen_feats[gen + '__toks']
            out[gen] = {'n': len(tk)}
            ps_store = {}
            for meth in methods:
                t_emb, m_emb = embed(gen_feats[gen][meth], tk)
                (r1, r2, r3), mmdist, ps = rprec_and_dist(t_emb, m_emb)
                ps_store[meth] = ps
                mu, cov = calculate_activation_statistics(m_emb)
                out[gen][meth] = {
                    'FID_vs_GT': float(calculate_frechet_distance(gt_mu, gt_cov, mu, cov)),
                    'R_top1': float(r1), 'R_top2': float(r2), 'R_top3': float(r3),
                    'MMDist': mmdist,
                    'Diversity': float(calculate_diversity(m_emb, min(100, len(m_emb) - 1))),
                }
            for meth in methods:
                if meth == 'original':
                    continue
                d, lo, hi, sig = paired_bootstrap(ps_store[meth], ps_store['original'])
                out[gen][meth]['dMMDist_vs_original'] = d
                out[gen][meth]['dMMDist_CI95'] = [lo, hi]
                out[gen][meth]['dMMDist_significant'] = sig
        return out
    finally:
        os.chdir(cwd)
        sys.path = saved_path
        for k in list(sys.modules):
            if k in ('utils', 'models', 'options') or k.startswith(('utils.', 'models.', 'options.')):
                del sys.modules[k]
        sys.modules.update(saved_mods)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v19', default='checkpoints/v19_088a10/best.pth')
    ap.add_argument('--n-ref', type=int, default=0, help='cap reference size (0 = all local)')
    ap.add_argument('--n-gen', type=int, default=0, help='cap per-generator samples (0 = all)')
    args = ap.parse_args()

    log("=" * 88)
    log("  LARGE-REFERENCE FID / SEMANTIC  (fixes n=26 reference)")
    log("=" * 88)

    ref_feats, ref_toks = build_reference(args.n_ref)
    gen_feats, methods = build_generated(args)
    res = evaluate(ref_feats, ref_toks, gen_feats, methods)

    with open(f"{ANA}/semantic_largeref.json", 'w') as f:
        json.dump(res, f, indent=2)

    log("")
    log(f"  FID reference n = {res['_fid_reference_n']}   embedding dim = {res['_embedding_dim']}")
    for gen in GENERATORS:
        r = res[gen]
        log("")
        log(f"── {gen.upper()}  (n={r['n']}) ──")
        log(f"  {'method':<12} {'FID↓':>8} {'R@1↑':>7} {'R@2↑':>7} {'R@3↑':>7} "
            f"{'MMDist↓':>8} {'Divers':>7}")
        for m in methods:
            v = r[m]
            log(f"  {m:<12} {v['FID_vs_GT']:8.3f} {v['R_top1']:7.3f} {v['R_top2']:7.3f} "
                f"{v['R_top3']:7.3f} {v['MMDist']:8.3f} {v['Diversity']:7.3f}")
        log(f"  -- MMDist change vs original (paired bootstrap, 95% CI; + = worse) --")
        for m in methods:
            if m == 'original':
                continue
            v = r[m]
            lo, hi = v['dMMDist_CI95']
            mark = "SIGNIFICANT" if v['dMMDist_significant'] else "n.s."
            log(f"     {m:<12} {v['dMMDist_vs_original']:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {mark}")
    log("")
    log(f"  → {ANA}/semantic_largeref.json")


if __name__ == "__main__":
    main()
