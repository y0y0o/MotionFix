"""
V15 — Physics-Teacher Test
============================
Tests V15 on 10 held-out MoMask motions:
  - V15 model output (trained to replicate physics correction)
  - Physics fix (teacher reference)
  - MoMask original (baseline)

Three-way comparison with full 7-metric evaluation.
"""
import torch, numpy as np, pickle, os, sys, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.v14 import MotionFixNetworkV14
from utils.physics_fix import physics_foot_fix
from utils.metrics import (
    compute_fsr, compute_jitter, compute_floating, compute_foot_error,
    compute_bone_length_consistency, compute_ground_penetration,
    compute_contact_accuracy,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_PATH = "data/training/v15/test_data.pkl"
OUTPUT_DIR = "outputs/fixed/v15"
LOG_PATH = "logs/v15_test.log"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg, also_print=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')
    if also_print:
        print(line)


def infer_v15(model, motion_world):
    """V15 inference: world coords → root-relative → model → world."""
    T = motion_world.shape[0]
    flat = motion_world.reshape(T, -1).astype(np.float32)
    tensor = torch.from_numpy(flat).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        result = model(tensor, foot_only=True, root_relative=True)
    return result.squeeze(0).cpu().numpy().reshape(T, 22, 3)


def compute_all(motion, original):
    return {
        'FSR': compute_fsr(motion)[0],
        'Jitter': compute_jitter(motion),
        'Floating': compute_floating(motion)[0],
        'FootErr': compute_foot_error(motion, original),
        'ContactAcc': compute_contact_accuracy(motion, original),
        'BoneCV': compute_bone_length_consistency(motion),
        'PenetrationMean': compute_ground_penetration(motion)[0],
        'PenetrationMax': compute_ground_penetration(motion)[1],
    }


def main():
    log("=" * 70)
    log("  V15 — Physics-Teacher Test (10 held-out motions)")
    log("=" * 70)

    # Load model
    ckpt = torch.load("checkpoints/v15/best.pth", map_location=DEVICE)
    model = MotionFixNetworkV14(blend_alpha=0.5).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    log(f"  Model: V15 (epoch {ckpt['epoch']+1}, loss {ckpt['loss']:.4f})")

    # Load test data
    with open(TEST_PATH, 'rb') as f:
        test_data = pickle.load(f)
    log(f"  Test motions: {len(test_data)}")
    log("")

    # Header
    hdr = (f"{'#':>2} {'Name':<35} {'Orig':>8} {'Phys':>8} {'V15':>8} "
           f"{'ΔV15':>6} {'ΔPhys':>6} {'Jit_orig':>8} {'Jit_V15':>8}")
    log(hdr)
    log("-" * len(hdr))

    results = {'original': [], 'physics': [], 'v15': []}

    for i, d in enumerate(test_data):
        orig_world = d['original_world']
        phys_world = d['target_world']

        # V15 inference
        v15_world = infer_v15(model, orig_world)

        # Compute metrics
        m_orig = compute_all(orig_world, orig_world)
        m_phys = compute_all(phys_world, orig_world)
        m_v15 = compute_all(v15_world, orig_world)

        # Save outputs
        name = d['name']
        np.save(f"{OUTPUT_DIR}/{name}_v15.npy", v15_world)

        log(f"{i+1:>2} {name[:34]:<35} "
            f"{m_orig['FSR']:>7.1%} {m_phys['FSR']:>7.1%} {m_v15['FSR']:>7.1%} "
            f"{m_v15['FSR']-m_orig['FSR']:>+5.1%} {m_phys['FSR']-m_orig['FSR']:>+5.1%} "
            f"{m_orig['Jitter']:>8.4f} {m_v15['Jitter']:>8.4f}")

        results['original'].append(m_orig)
        results['physics'].append(m_phys)
        results['v15'].append(m_v15)

    log("-" * len(hdr))

    # Summary
    def avg_metrics(metrics_list):
        return {k: float(np.mean([m[k] for m in metrics_list]))
                for k in metrics_list[0].keys()}

    a_orig = avg_metrics(results['original'])
    a_phys = avg_metrics(results['physics'])
    a_v15 = avg_metrics(results['v15'])

    log("")
    log("=" * 70)
    log(f"  📊 V15 vs Physics vs Original (n={len(test_data)})")
    log("=" * 70)
    log(f"  {'Metric':<20} {'Original':>10} {'Physics':>10} {'V15':>10} {'V15 vs Phys':>12}")
    log(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")

    for key, label in [('FSR', 'FSR'), ('Jitter', 'Jitter'),
                        ('Floating', 'Floating'), ('FootErr', 'Foot Error'),
                        ('ContactAcc', 'Contact Acc'), ('BoneCV', 'Bone CV')]:
        o = a_orig[key]; p = a_phys[key]; v = a_v15[key]
        if key in ('FSR', 'Floating', 'ContactAcc'):
            log(f"  {label:<20} {o:>9.1%} {p:>9.1%} {v:>9.1%} "
                f"{'→' + str(abs(v-p)) + 'diff':>12}")
        else:
            log(f"  {label:<20} {o:>10.4f} {p:>10.4f} {v:>10.4f} "
                f"{'Δ='+str(v-p)[:6]:>12}")

    # Key question: does V15 match or beat the physics teacher?
    log(f"")
    if a_v15['FSR'] <= a_phys['FSR'] * 1.05:
        log(f"  ✅ V15 FSR ({a_v15['FSR']:.1%}) matches physics teacher ({a_phys['FSR']:.1%})")
    else:
        log(f"  ❌ V15 FSR ({a_v15['FSR']:.1%}) worse than physics ({a_phys['FSR']:.1%})")

    if a_v15['Jitter'] < a_phys['Jitter']:
        log(f"  ✅ V15 Jitter ({a_v15['Jitter']:.4f}) LOWER than physics ({a_phys['Jitter']:.4f}) — smoother!")
    else:
        log(f"  ⚠️  V15 Jitter ({a_v15['Jitter']:.4f}) ≥ physics ({a_phys['Jitter']:.4f})")

    log("=" * 70)
    log(f"  💾 Outputs: {OUTPUT_DIR}/")
    log(f"  📝 Log: {LOG_PATH}")
    log("  ✅ Done.")


if __name__ == "__main__":
    main()
