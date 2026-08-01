"""
Semantic-preservation evaluation — FID / R-precision / MM-Dist / Diversity
=========================================================================
Answers the one question every physics metric in `utils/metrics.py` is blind to,
and the one the supervisor asked for on 2026-06-24 and never got:

    after we move the feet, does the motion still MEAN what the text asked for?

Protocol
--------
* Text-motion evaluator: HumanML3D `text_mot_match` (the standard one used by
  MoMask / MDM / T2M-GPT papers), loaded from the T2M-GPT checkout.
* 50 motions per generator; prompts from `HumanML3D/test_prompts_50.txt`, POS
  tokens from `HumanML3D/texts/<id>.txt`.
* Every method — including the untouched original AND the ground truth — is
  pushed through the SAME `joints_to_263` conversion, so the conversion's own
  error is common-mode (see `utils/joints_to_feats.py` for why that matters).
* FID is measured against the HumanML3D ground truth for the same 50 IDs.

Reading the numbers
-------------------
* **MM-Dist** (text<->motion embedding distance) and **R-precision** are the
  direct answers: they degrade if the correction breaks text alignment.
* **FID** at n=50 is noisy in absolute terms; read the ORDERING, not the value.
* Absolute values are NOT comparable to published numbers (different conversion
  path, n=50 instead of the full test set).

Usage:
    python testing/v19_semantic.py --v19 checkpoints/v19_045/best.pth
Output:
    analysis/v19/semantic.json,  logs/v19_semantic.log
"""
import os, sys, json, glob, argparse
from datetime import datetime

import numpy as np
import torch

_MF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_T2M = "/home3/nxkh91/projects/T2M-GPT"
_HML = "/home3/nxkh91/projects/HumanML3D"

sys.path.insert(0, _MF)
# ── phase 1 imports: motionfix's own packages ──────────────────────────────
from utils.joints_to_feats import joints_to_263                    # noqa: E402
from models.v18 import FootRefiner, smooth_fix, deskate_xz, FOOT_XZ_DIMS  # noqa: E402
from models.v18_ik import apply_ik                                 # noqa: E402
from models.v19 import V19Smoother, smooth_fix_v19                 # noqa: E402

from scipy.ndimage import gaussian_filter1d                        # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GENERATORS = {
    'momask': ("data/test_inputs/momask_50/momask_50_results/no_ik", "p{id}_"),
    'mdm':    ("data/test_inputs/mdm/mdm_raw_joints", "mdm_{id}_"),
    't2mgpt': ("data/test_inputs/t2mgpt/t2mgpt_raw_joints", "t2mgpt_{id}_"),
}
PROMPTS = f"{_HML}/HumanML3D/test_prompts_50.txt"
TEXTS = f"{_HML}/HumanML3D/texts"
GT_VECS = f"{_HML}/HumanML3D/new_joint_vecs"
V18_CKPT = "checkpoints/v18_ik/best.pth"
GAUSS_SIGMA = 1.5
ANA, LOG = "analysis/v19", "logs/v19_semantic.log"
MAX_LEN, UNIT = 196, 4
N_RPREC_POOL, N_RPREC_REPEAT = 32, 200
os.makedirs(ANA, exist_ok=True); os.makedirs(os.path.dirname(LOG), exist_ok=True)


def log(m):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}"
    with open(LOG, 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


# ══════════════════════════════════════════════════════════════════════════
# Phase 1 — build corrected motions and convert them to 263-d features
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


def read_prompts():
    """id -> (caption, tokens) using the FIRST annotation of each motion."""
    ids, caps, toks = [], [], []
    for line in open(PROMPTS):
        line = line.strip()
        if not line:
            continue
        mid = line.split('|')[0]
        tf = f"{TEXTS}/{mid}.txt"
        if not os.path.exists(tf):
            continue
        first = open(tf).readline().strip().split('#')
        ids.append(mid); caps.append(first[0]); toks.append(first[1].split(' '))
    return ids, caps, toks


def build_features(args):
    """Returns feats[gen][method] = list of (T,263) arrays, aligned with ids."""
    ids, caps, toks = read_prompts()
    log(f"  prompts: {len(ids)} motions with POS tokens")

    m18 = FootRefiner().to(DEVICE)
    m18.load_state_dict(torch.load(V18_CKPT, map_location=DEVICE)['model_state_dict'])
    m18.eval()
    m19 = None
    if args.v19:
        m19 = V19Smoother().to(DEVICE)
        m19.load_state_dict(torch.load(args.v19, map_location=DEVICE)['model_state_dict'])
        m19.eval()

    methods = ['original', 'deskate_ik', 'gauss_ik', 'learn_ik'] + (['v19_ik'] if m19 else [])
    feats, kept_ids = {}, None

    # ── ground truth, through the SAME conversion path ──
    # `utils` is already bound to motionfix's package, so T2M-GPT's utils.* cannot
    # be imported by name here. Load the two files it needs directly by path.
    import importlib.util

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _q = _load('t2m_quaternion', f"{_T2M}/utils/quaternion.py")
    _mp_src = open(f"{_T2M}/utils/motion_process.py").read().replace(
        "from utils.quaternion import", "from t2m_quaternion import")
    _mp = importlib.util.module_from_spec(
        importlib.util.spec_from_loader('t2m_motion_process', loader=None))
    exec(compile(_mp_src, 'motion_process.py', 'exec'), _mp.__dict__)
    recover_from_ric = _mp.recover_from_ric

    gt_feats, gt_ids = [], []
    for mid in ids:
        fp = f"{GT_VECS}/{mid}.npy"
        if not os.path.exists(fp):
            continue
        j = recover_from_ric(torch.from_numpy(np.load(fp)).float(), 22).numpy()
        gt_feats.append(joints_to_263(j[:MAX_LEN])); gt_ids.append(mid)
    feats['__gt__'] = {'gt': gt_feats}
    log(f"  ground truth converted: {len(gt_feats)}")

    for gen, (d, pat) in GENERATORS.items():
        acc = {k: [] for k in methods}
        gen_ids = []
        for mid in ids:
            hits = glob.glob(os.path.join(d, pat.format(id=mid) + "*.npy"))
            if not hits:
                continue
            mo = load_joints(hits[0])[:MAX_LEN]
            try:
                acc['original'].append(joints_to_263(mo))
                acc['deskate_ik'].append(joints_to_263(apply_ik(mo, deskate_only(mo, 0.0))))
                acc['gauss_ik'].append(joints_to_263(apply_ik(mo, deskate_only(mo, GAUSS_SIGMA))))
                acc['learn_ik'].append(joints_to_263(apply_ik(mo, smooth_fix(mo, m18, DEVICE))))
                if m19:
                    acc['v19_ik'].append(joints_to_263(apply_ik(mo, smooth_fix_v19(mo, m19, DEVICE))))
            except Exception as e:                     # noqa: BLE001
                log(f"    !! {gen}/{mid} conversion failed ({e}) — dropped")
                for k in acc:
                    if len(acc[k]) > len(gen_ids):
                        acc[k].pop()
                continue
            gen_ids.append(mid)
        feats[gen] = acc
        feats[gen + '__ids'] = gen_ids
        log(f"  {gen}: {len(gen_ids)} motions x {len(methods)} methods converted")
        kept_ids = gen_ids if kept_ids is None else kept_ids

    return feats, ids, caps, toks, methods


# ══════════════════════════════════════════════════════════════════════════
# Phase 2 — evaluator (runs entirely inside the T2M-GPT import context)
# ══════════════════════════════════════════════════════════════════════════

def evaluate(feats, ids, caps, toks, methods):
    cwd = os.getcwd()
    saved_mods = {k: v for k, v in sys.modules.items()
                  if k in ('utils', 'models', 'options') or
                  k.startswith(('utils.', 'models.', 'options.'))}
    for k in saved_mods:
        del sys.modules[k]
    saved_path = list(sys.path)
    sys.path = [p for p in sys.path if p not in ('', _MF)]
    sys.path.insert(0, _T2M)
    os.chdir(_T2M)                       # evaluator resolves ./checkpoints relatively

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
            m, ml = motion_batch(fl)
            w, p, cl = text_batch(tk)
            # TextEncoderBiGRUCo packs with enforce_sorted=True, so cap_lens must be
            # DESCENDING (the reference collate_fn sorts by it). The motion encoder
            # uses enforce_sorted=False and needs no ordering. Sort both by cap_len
            # so pairs stay aligned, then invert to restore the caller's order.
            order = torch.argsort(cl, descending=True)
            inv = torch.argsort(order)
            t_emb, m_emb = ev.get_co_embeddings(
                w[order], p[order], cl[order], m[order], ml[order])
            return (t_emb.cpu().numpy()[inv.numpy()],
                    m_emb.cpu().numpy()[inv.numpy()])

        def per_sample_mmdist(t_emb, m_emb):
            """Paired text<->motion distance, one value per motion."""
            return np.linalg.norm(t_emb - m_emb, axis=1)

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
            ps = per_sample_mmdist(t_emb, m_emb)
            return np.mean(tops, 0), float(ps.mean()), ps

        def paired_bootstrap(a, b, n_boot=10000, seed=0):
            """
            95% CI on mean(a - b) by paired bootstrap. `a`,`b` are per-sample
            MMDist for the same motions, so the pairing removes between-motion
            variance — the only way n=50 can say anything at all here.
            """
            d = a - b
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(d), (n_boot, len(d)))
            boot = d[idx].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            return float(d.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)

        # ground truth reference statistics
        gt_f = feats['__gt__']['gt']
        gt_tok = toks[:len(gt_f)]
        _, gt_m = embed(gt_f, gt_tok)
        gt_mu, gt_cov = calculate_activation_statistics(gt_m)

        out = {}
        for gen in GENERATORS:
            gen_ids = feats[gen + '__ids']
            tk = [toks[ids.index(i)] for i in gen_ids]
            out[gen] = {'n': len(gen_ids)}
            ps_store = {}
            for meth in methods:
                t_emb, m_emb = embed(feats[gen][meth], tk)
                (r1, r2, r3), mmdist, ps = rprec_and_dist(t_emb, m_emb)
                ps_store[meth] = ps
                mu, cov = calculate_activation_statistics(m_emb)
                out[gen][meth] = {
                    'FID_vs_GT': float(calculate_frechet_distance(gt_mu, gt_cov, mu, cov)),
                    'R_top1': float(r1), 'R_top2': float(r2), 'R_top3': float(r3),
                    'MMDist': mmdist,
                    'Diversity': float(calculate_diversity(m_emb, min(100, len(m_emb) - 1))),
                }
            # paired bootstrap: is the MMDist change vs `original` real at n=50?
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
    ap.add_argument('--v19', default='checkpoints/v19_045/best.pth')
    args = ap.parse_args()

    log("=" * 88)
    log("  SEMANTIC PRESERVATION — FID / R-precision / MM-Dist / Diversity")
    log("=" * 88)

    feats, ids, caps, toks, methods = build_features(args)
    res = evaluate(feats, ids, caps, toks, methods)

    with open(f"{ANA}/semantic.json", 'w') as f:
        json.dump(res, f, indent=2)

    for gen, r in res.items():
        log("")
        log(f"── {gen.upper()}  (n={r['n']}) ──")
        log(f"  {'method':<12} {'FID↓':>8} {'R@1↑':>7} {'R@2↑':>7} {'R@3↑':>7} "
            f"{'MMDist↓':>8} {'Divers':>7}")
        for m in methods:
            v = r[m]
            log(f"  {m:<12} {v['FID_vs_GT']:8.3f} {v['R_top1']:7.3f} {v['R_top2']:7.3f} "
                f"{v['R_top3']:7.3f} {v['MMDist']:8.3f} {v['Diversity']:7.3f}")
        log(f"  -- MMDist change vs original (paired bootstrap, 95% CI; + = worse alignment) --")
        for m in methods:
            if m == 'original':
                continue
            v = r[m]
            lo, hi = v['dMMDist_CI95']
            mark = "SIGNIFICANT" if v['dMMDist_significant'] else "n.s."
            log(f"     {m:<12} {v['dMMDist_vs_original']:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {mark}")
    log("")
    log(f"  → {ANA}/semantic.json")


if __name__ == "__main__":
    main()
