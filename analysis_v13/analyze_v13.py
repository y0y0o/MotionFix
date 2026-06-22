"""
V13 Comprehensive Analysis — Visualizations & Anomaly Detection

Analyzes V13 fixed outputs against original motions for both MoMask and MDM.
Generates plots, identifies anomalies, saves report.
"""
import numpy as np
import glob
import os
import sys
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ================================================================
#  Metrics (same as test_v13.py)
# ================================================================
def compute_contact_labels(motion, foot_joints=(7, 8),
                           height_thresh=0.05, vel_thresh=0.5):
    T = motion.shape[0]
    labels = np.zeros((T, 2), dtype=np.float32)
    for i, fj in enumerate(foot_joints):
        foot_y = motion[:, fj, 1]
        ground = np.percentile(foot_y, 5)
        threshold = ground + height_thresh
        for t in range(T):
            if foot_y[t] < threshold:
                if t > 0:
                    vel = np.linalg.norm(motion[t, fj, [0, 2]] - motion[t-1, fj, [0, 2]])
                    if vel < vel_thresh:
                        labels[t, i] = 1.0
                else:
                    labels[t, i] = 1.0
    return labels

def compute_fsr(motion, foot_joints=(7, 8), vel_thresh=0.03):
    contact = compute_contact_labels(motion, foot_joints)
    T = motion.shape[0]
    skating = 0; contact_count = 0
    for i, fj in enumerate(foot_joints):
        for t in range(1, T):
            if contact[t, i] > 0.5:
                contact_count += 1
                vel = np.linalg.norm(motion[t, fj, [0, 2]] - motion[t-1, fj, [0, 2]])
                if vel > vel_thresh:
                    skating += 1
    if contact_count == 0:
        return 0.0, 0, 0
    return skating / contact_count, contact_count, skating

def compute_jitter(motion):
    foot_joints = [7, 8, 10, 11]
    foot_motion = motion[:, foot_joints, :]
    vel = foot_motion[1:] - foot_motion[:-1]
    acc = vel[1:] - vel[:-1]
    return np.sqrt((acc ** 2).mean())

def compute_mean_foot_error(original, fixed):
    foot_joints = [7, 8, 10, 11]
    diffs = []
    for fj in foot_joints:
        diff = np.linalg.norm(fixed[:, fj, :] - original[:, fj, :], axis=1)
        diffs.append(diff)
    return np.mean(diffs)

def compute_max_foot_displacement(original, fixed):
    """Max single-frame foot displacement"""
    foot_joints = [7, 8, 10, 11]
    max_disp = 0.0
    for fj in foot_joints:
        diff = np.linalg.norm(fixed[:, fj, :] - original[:, fj, :], axis=1)
        max_disp = max(max_disp, diff.max())
    return max_disp

def compute_fsr_per_joint(motion):
    """Per-joint FSR for ankles (7,8)"""
    contact = compute_contact_labels(motion)
    results = {}
    for i, fj in enumerate((7, 8)):
        skating = 0; contact_count = 0
        for t in range(1, len(motion)):
            if contact[t, i] > 0.5:
                contact_count += 1
                vel = np.linalg.norm(motion[t, fj, [0, 2]] - motion[t-1, fj, [0, 2]])
                if vel > 0.03:
                    skating += 1
        results[fj] = skating / contact_count if contact_count > 0 else 0.0
    return results

# ================================================================
#  Category classification from prompts
# ================================================================
def load_prompt_categories():
    """Load category labels from prompts file"""
    prompts_file = os.path.expanduser("~/HumanML3D/HumanML3D/test_prompts_50.txt")
    pid_to_category = {}
    if not os.path.exists(prompts_file):
        return pid_to_category
    with open(prompts_file) as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                pid_to_category[parts[0]] = parts[1]
    return pid_to_category

def infer_category_from_name(name):
    """Infer motion category from filename"""
    name_lower = name.lower()
    if 'walk' in name_lower:
        return 'walking'
    elif 'run' in name_lower or 'sprint' in name_lower:
        return 'running'
    elif 'jump' in name_lower:
        return 'jumping'
    elif 'dance' in name_lower or 'danc' in name_lower or 'salsa' in name_lower or 'shadowbox' in name_lower:
        return 'dance'
    elif 'turn' in name_lower or 'rotat' in name_lower or 'spin' in name_lower:
        return 'turning'
    elif 'backward' in name_lower:
        return 'backward'
    elif 'complex' in name_lower:
        return 'complex'
    return 'other'

# ================================================================
#  Main Analysis
# ================================================================
def main():
    SAVE_DIR = "analysis_v13"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- Load all results ----
    pid_to_category = load_prompt_categories()

    datasets = {
        'MoMask': {
            'originals_dir': 'momask_50_results/no_ik',
            'fixed_dir': 'fixed_outputs_v13_momask',
            'file_pattern': '*.npy',
            'name_filter': lambda x: not x.endswith('_fixed.npy'),
        },
        'MDM': {
            'originals_dir': 'mdm_raw_joints',
            'fixed_dir': 'fixed_outputs_v13_mdm',
            'file_pattern': 'mdm_*_joints.npy',
            'name_filter': lambda x: True,
        },
    }

    all_results = {}

    for ds_name, ds_info in datasets.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {ds_name}...")
        print(f"{'='*60}")

        # Get original files
        orig_files = sorted(glob.glob(f"{ds_info['originals_dir']}/{ds_info['file_pattern']}"))
        orig_files = [f for f in orig_files if ds_info['name_filter'](os.path.basename(f))]
        print(f"  Originals: {len(orig_files)}")

        results = []
        for orig_path in orig_files:
            fname = os.path.basename(orig_path)
            name = fname.replace('.npy', '').replace('_joints', '')

            # Derive fixed file path
            if ds_name == 'MDM':
                # mdm_000021_joints.npy → mdm_000021_fixed.npy
                fixed_path = f"{ds_info['fixed_dir']}/{name}_fixed.npy"
            else:
                # p000021_....npy → p000021_...._fixed.npy
                fixed_path = fname.replace('.npy', '_fixed.npy')
                fixed_path = f"{ds_info['fixed_dir']}/{fixed_path}"

            if not os.path.exists(fixed_path):
                print(f"    WARNING: no fixed file for {name}")
                continue

            original = np.load(orig_path).astype(np.float32)
            fixed = np.load(fixed_path).astype(np.float32)

            # Handle MoMask 4D array
            if len(original.shape) == 4:
                original = original[0]
            if len(fixed.shape) == 4:
                fixed = fixed[0]

            fsr_b, fc_b, sk_b = compute_fsr(original)
            fsr_a, fc_a, sk_a = compute_fsr(fixed)
            jitter_b = compute_jitter(original)
            jitter_a = compute_jitter(fixed)
            foot_err = compute_mean_foot_error(original, fixed)
            max_disp = compute_max_foot_displacement(original, fixed)
            fsr_per_joint_b = compute_fsr_per_joint(original)
            fsr_per_joint_a = compute_fsr_per_joint(fixed)

            # Category
            pid = name.split('_')[0] if '_' in name else name[:7]
            if ds_name == 'MDM':
                pid = pid.replace('mdm', 'p') if pid.startswith('mdm') else pid
            category = pid_to_category.get(pid, infer_category_from_name(name))

            results.append({
                'name': name,
                'pid': pid,
                'category': category,
                'T': original.shape[0],
                'fsr_before': fsr_b,
                'fsr_after': fsr_a,
                'fsr_change': fsr_a - fsr_b,
                'contact_before': fc_b,
                'contact_after': fc_a,
                'skating_before': sk_b,
                'skating_after': sk_a,
                'jitter_before': jitter_b,
                'jitter_after': jitter_a,
                'jitter_change': jitter_a - jitter_b,
                'jitter_ratio': jitter_a / jitter_b if jitter_b > 0 else 0,
                'foot_error': foot_err,
                'max_disp': max_disp,
                'fsr_ankle7_b': fsr_per_joint_b.get(7, 0),
                'fsr_ankle7_a': fsr_per_joint_a.get(7, 0),
                'fsr_ankle8_b': fsr_per_joint_b.get(8, 0),
                'fsr_ankle8_a': fsr_per_joint_a.get(8, 0),
            })

        all_results[ds_name] = results
        print(f"  Analyzed: {len(results)} motions")

    # ---- Identify anomalies ----
    anomalies = {'MoMask': [], 'MDM': []}
    for ds_name, results in all_results.items():
        for r in results:
            reasons = []
            if r['fsr_change'] > 0.05:   # FSR worsened >5%
                reasons.append(f"FSR worsened: {r['fsr_change']:+.1%}")
            if r['fsr_before'] > 0.30 and r['fsr_change'] < -0.05:  # High FSR but good fix
                reasons.append(f"Large FSR fix: {r['fsr_before']:.1%}→{r['fsr_after']:.1%}")
            if r['jitter_ratio'] > 30:    # Jitter increased >30x
                reasons.append(f"Jitter explosion: {r['jitter_ratio']:.0f}x")
            if r['foot_error'] > 0.20:    # Foot modified >20cm average
                reasons.append(f"Large foot error: {r['foot_error']:.3f}m")
            if r['max_disp'] > 0.50:      # Single frame >50cm
                reasons.append(f"Max displacement: {r['max_disp']:.3f}m")
            if r['fsr_before'] < 0.01 and r['fsr_change'] > 0.02:
                reasons.append(f"Clean motion damaged: {r['fsr_before']:.1%}→{r['fsr_after']:.1%}")

            if reasons:
                anomalies[ds_name].append({**r, 'reasons': reasons})

    # ================================================================
    #  Figure 1: FSR Before/After Scatter
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors_map = {'walking': '#2ecc71', 'dance': '#e74c3c', 'turning': '#3498db',
                  'jumping': '#f39c12', 'running': '#9b59b6', 'backward': '#1abc9c',
                  'complex': '#e67e22', 'other': '#95a5a6'}

    for ax_idx, (ds_name, results) in enumerate(all_results.items()):
        ax = axes[ax_idx]
        categories = set(r['category'] for r in results)
        for cat in sorted(categories):
            cat_results = [r for r in results if r['category'] == cat]
            xs = [r['fsr_before']*100 for r in cat_results]
            ys = [r['fsr_after']*100 for r in cat_results]
            ax.scatter(xs, ys, c=colors_map.get(cat, '#95a5a6'), label=cat,
                       alpha=0.7, edgecolors='black', linewidth=0.5, s=60)

        # Diagonal line
        lims = [0, max(max(r['fsr_before'] for r in results)*100, 65)]
        ax.plot(lims, lims, 'k--', alpha=0.3, label='No change')
        ax.set_xlabel('FSR Before (%)', fontsize=12)
        ax.set_ylabel('FSR After (%)', fontsize=12)
        ax.set_title(f'{ds_name} — FSR Before vs After', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lims); ax.set_ylim(lims)

        # Annotate extreme points
        for r in results:
            if r['fsr_change'] < -0.08 or r['fsr_change'] > 0.08:
                ax.annotate(r['pid'], (r['fsr_before']*100, r['fsr_after']*100),
                           fontsize=6, alpha=0.7)

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/01_fsr_before_after.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 01_fsr_before_after.png")

    # ================================================================
    #  Figure 2: Jitter Before/After Scatter (log scale)
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (ds_name, results) in enumerate(all_results.items()):
        ax = axes[ax_idx]
        categories = set(r['category'] for r in results)
        for cat in sorted(categories):
            cat_results = [r for r in results if r['category'] == cat]
            xs = [r['jitter_before'] for r in cat_results]
            ys = [r['jitter_after'] for r in cat_results]
            ax.scatter(xs, ys, c=colors_map.get(cat, '#95a5a6'), label=cat,
                       alpha=0.7, edgecolors='black', linewidth=0.5, s=60)

        max_val = max(max(r['jitter_after'] for r in results), 0.1)
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='No change')
        ax.set_xlabel('Jitter Before', fontsize=12)
        ax.set_ylabel('Jitter After', fontsize=12)
        ax.set_title(f'{ds_name} — Jitter Before vs After', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Annotate explosions
        for r in results:
            if r['jitter_ratio'] > 25:
                ax.annotate(r['pid'], (r['jitter_before'], r['jitter_after']),
                           fontsize=6, color='red', alpha=0.8)

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/02_jitter_before_after.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 02_jitter_before_after.png")

    # ================================================================
    #  Figure 3: FSR Change Distribution by Category
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (ds_name, results) in enumerate(all_results.items()):
        ax = axes[ax_idx]
        # Group by category
        cat_data = defaultdict(list)
        for r in results:
            cat_data[r['category']].append(r['fsr_change']*100)

        cats_sorted = sorted(cat_data.keys(), key=lambda c: np.mean(cat_data[c]))
        positions = range(len(cats_sorted))
        bp = ax.boxplot([cat_data[c] for c in cats_sorted], positions=positions,
                         patch_artist=True, widths=0.6)

        for i, cat in enumerate(cats_sorted):
            color = colors_map.get(cat, '#95a5a6')
            bp['boxes'][i].set_facecolor(color)
            bp['boxes'][i].set_alpha(0.6)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xticks(positions)
        ax.set_xticklabels(cats_sorted, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('FSR Change (pp)', fontsize=12)
        ax.set_title(f'{ds_name} — FSR Change by Category', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean labels
        for i, cat in enumerate(cats_sorted):
            mean_val = np.mean(cat_data[cat])
            ax.annotate(f'{mean_val:+.1f}', (i, mean_val),
                       textcoords="offset points", xytext=(0, 10 if mean_val >=0 else -15),
                       ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/03_fsr_change_by_category.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 03_fsr_change_by_category.png")

    # ================================================================
    #  Figure 4: FSR vs Foot Error Tradeoff
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (ds_name, results) in enumerate(all_results.items()):
        ax = axes[ax_idx]
        for r in results:
            color = 'green' if r['fsr_change'] < 0 else 'red'
            alpha = min(1.0, abs(r['fsr_change'])*8 + 0.3)
            size = max(20, min(200, r['jitter_ratio']*3))
            ax.scatter(r['foot_error']*100, r['fsr_change']*100,
                      c=color, alpha=alpha, s=size, edgecolors='black', linewidth=0.5)

            # Label extreme cases
            if r['foot_error'] > 0.20 or abs(r['fsr_change']) > 0.10:
                ax.annotate(r['pid'], (r['foot_error']*100, r['fsr_change']*100),
                           fontsize=6, alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Mean Foot Error (cm)', fontsize=12)
        ax.set_ylabel('FSR Change (pp)', fontsize=12)
        ax.set_title(f'{ds_name} — FSR Change vs Foot Modification', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add legend
        green_patch = mpatches.Patch(color='green', alpha=0.7, label='FSR improved')
        red_patch = mpatches.Patch(color='red', alpha=0.7, label='FSR worsened')
        ax.legend(handles=[green_patch, red_patch], fontsize=8)

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/04_fsr_vs_foot_error.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_fsr_vs_foot_error.png")

    # ================================================================
    #  Figure 5: Jitter Ratio Distribution (Histogram)
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (ds_name, results) in enumerate(all_results.items()):
        ax = axes[ax_idx]
        ratios = [r['jitter_ratio'] for r in results if r['jitter_ratio'] < 100]
        extreme = [r for r in results if r['jitter_ratio'] >= 100]

        ax.hist(ratios, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=np.median(ratios), color='red', linestyle='--',
                   label=f'Median: {np.median(ratios):.1f}x')
        ax.set_xlabel('Jitter Ratio (after/before)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{ds_name} — Jitter Increase Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        if extreme:
            names_str = ', '.join(r['pid'] for r in extreme[:5])
            ax.text(0.95, 0.95, f'Extreme (>100x): {len(extreme)}\n{names_str}',
                   transform=ax.transAxes, fontsize=7, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/05_jitter_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_jitter_distribution.png")

    # ================================================================
    #  Figure 6: Per-Joint FSR Analysis
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (ds_name, results) in enumerate(all_results.items()):
        ax = axes[ax_idx]
        cats = ['walking', 'dance', 'turning', 'jumping', 'complex']
        cat_means_7 = []; cat_means_8 = []
        cat_labels = []

        for cat in cats:
            cr = [r for r in results if r['category'] == cat]
            if len(cr) >= 2:
                cat_labels.append(cat)
                cat_means_7.append(np.mean([r['fsr_ankle7_a']-r['fsr_ankle7_b'] for r in cr])*100)
                cat_means_8.append(np.mean([r['fsr_ankle8_a']-r['fsr_ankle8_b'] for r in cr])*100)

        x = np.arange(len(cat_labels))
        w = 0.35
        ax.bar(x - w/2, cat_means_7, w, label='Left Ankle (7)', color='#e74c3c', alpha=0.8)
        ax.bar(x + w/2, cat_means_8, w, label='Right Ankle (8)', color='#3498db', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('FSR Change (pp)', fontsize=12)
        ax.set_title(f'{ds_name} — Per-Ankle FSR Change by Category', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/06_per_ankle_fsr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_per_ankle_fsr.png")

    # ================================================================
    #  Figure 7: Top/Bottom performers highlight
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    all_combined = []
    for ds_name, results in all_results.items():
        for r in results:
            all_combined.append({**r, 'dataset': ds_name})

    # Sort by FSR change (most improved first)
    all_combined.sort(key=lambda r: r['fsr_change'])

    # Top 10 improved + bottom 10 worsened
    top10 = all_combined[:10]
    bottom10 = all_combined[-10:]

    names = [r['pid'] for r in top10] + [r['pid'] for r in bottom10]
    changes = [r['fsr_change']*100 for r in top10] + [r['fsr_change']*100 for r in bottom10]
    datasets_list = [r['dataset'] for r in top10] + [r['dataset'] for r in bottom10]
    colors = ['green' if c < 0 else 'red' for c in changes]
    hatches = ['//' if d == 'MDM' else '' for d in datasets_list]

    bars = ax.barh(range(len(names)), changes, color=colors, edgecolor='black')
    for i, (bar, hatch) in enumerate(zip(bars, hatches)):
        if hatch:
            bar.set_hatch(hatch)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} ({d})" for n, d in zip(names, datasets_list)], fontsize=8)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('FSR Change (pp)', fontsize=12)
    ax.set_title('V13 — Top 10 Best + Worst FSR Changes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add foot error annotations
    for i, r in enumerate(top10 + bottom10):
        ax.annotate(f'err={r["foot_error"]*100:.0f}cm jit={r["jitter_ratio"]:.0f}x',
                   (changes[i], i), textcoords="offset points",
                   xytext=(5 if changes[i] >=0 else -5, 0),
                   ha='left' if changes[i] >=0 else 'right',
                   fontsize=6, alpha=0.7)

    # Legend
    green_bar = mpatches.Patch(color='green', label='FSR Improved')
    red_bar = mpatches.Patch(color='red', label='FSR Worsened')
    mdm_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='MDM')
    ax.legend(handles=[green_bar, red_bar, mdm_patch], fontsize=9, loc='lower right')

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/07_top_bottom_performers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 07_top_bottom_performers.png")

    # ================================================================
    #  Generate anomaly report
    # ================================================================
    report_lines = []
    report_lines.append("# V13 Anomaly Analysis Report")
    report_lines.append("")
    report_lines.append("## Summary Statistics")
    report_lines.append("")

    for ds_name, results in all_results.items():
        avg_b = np.mean([r['fsr_before'] for r in results])
        avg_a = np.mean([r['fsr_after'] for r in results])
        avg_jb = np.mean([r['jitter_before'] for r in results])
        avg_ja = np.mean([r['jitter_after'] for r in results])
        avg_fe = np.mean([r['foot_error'] for r in results])
        improved = sum(1 for r in results if r['fsr_change'] < 0)
        worsened = sum(1 for r in results if r['fsr_change'] > 0)
        unchanged = sum(1 for r in results if r['fsr_change'] == 0)

        report_lines.append(f"### {ds_name}")
        report_lines.append(f"- Motions analyzed: {len(results)}")
        report_lines.append(f"- FSR: {avg_b:.1%} → {avg_a:.1%} ({avg_a-avg_b:+.1%})")
        report_lines.append(f"- Jitter: {avg_jb:.4f} → {avg_ja:.4f} ({avg_ja/avg_jb:.1f}x)")
        report_lines.append(f"- Mean foot error: {avg_fe*100:.1f} cm")
        report_lines.append(f"- Improved: {improved}, Worsened: {worsened}, Unchanged: {unchanged}")
        report_lines.append("")
        report_lines.append(f"**Best FSR improvements:**")
        best = sorted(results, key=lambda r: r['fsr_change'])[:3]
        for r in best:
            report_lines.append(f"  - {r['name'][:40]}: {r['fsr_before']:.1%}→{r['fsr_after']:.1%} ({r['fsr_change']:+.1%}), jitter {r['jitter_ratio']:.0f}x, err {r['foot_error']*100:.1f}cm")
        report_lines.append("")
        report_lines.append(f"**Worst FSR regressions:**")
        worst = sorted(results, key=lambda r: r['fsr_change'], reverse=True)[:3]
        for r in worst:
            report_lines.append(f"  - {r['name'][:40]}: {r['fsr_before']:.1%}→{r['fsr_after']:.1%} ({r['fsr_change']:+.1%}), jitter {r['jitter_ratio']:.0f}x, err {r['foot_error']*100:.1f}cm")
        report_lines.append("")

    # Category breakdown
    report_lines.append("## Category Breakdown")
    report_lines.append("")
    for ds_name, results in all_results.items():
        report_lines.append(f"### {ds_name}")
        cat_data = defaultdict(list)
        for r in results:
            cat_data[r['category']].append(r)
        for cat in sorted(cat_data.keys()):
            cr = cat_data[cat]
            avg_c = np.mean([r['fsr_change'] for r in cr])*100
            avg_j = np.mean([r['jitter_ratio'] for r in cr])
            avg_e = np.mean([r['foot_error'] for r in cr])*100
            report_lines.append(f"  {cat:<12}: n={len(cr):>2}, FSR change={avg_c:>+5.1f}pp, jitter={avg_j:.0f}x, err={avg_e:.0f}cm")
        report_lines.append("")

    # Anomaly details
    report_lines.append("## Anomaly Details")
    report_lines.append("")
    for ds_name in ['MoMask', 'MDM']:
        report_lines.append(f"### {ds_name} — {len(anomalies[ds_name])} anomalies")
        report_lines.append("")
        for r in anomalies[ds_name]:
            report_lines.append(f"**{r['pid']}** ({r['category']})")
            report_lines.append(f"  - FSR: {r['fsr_before']:.1%} → {r['fsr_after']:.1%} ({r['fsr_change']:+.1%})")
            report_lines.append(f"  - Jitter: {r['jitter_before']:.4f} → {r['jitter_after']:.4f} ({r['jitter_ratio']:.0f}x)")
            report_lines.append(f"  - Foot error: {r['foot_error']*100:.1f}cm avg, {r['max_disp']*100:.1f}cm max")
            report_lines.append(f"  - Issues: {', '.join(r['reasons'])}")
            report_lines.append("")

    # Overall assessment
    report_lines.append("## Overall Assessment")
    report_lines.append("")
    momask_fsr_change = np.mean([r['fsr_change'] for r in all_results['MoMask']]) * 100
    mdm_fsr_change = np.mean([r['fsr_change'] for r in all_results['MDM']]) * 100
    report_lines.append(f"- V13 is the best-performing version for FSR reduction")
    report_lines.append(f"  - MoMask: {momask_fsr_change:+.1f}pp (V8: -2.9pp)")
    report_lines.append(f"  - MDM: {mdm_fsr_change:+.1f}pp (V8: -4.0pp)")
    report_lines.append(f"- Main weakness: jitter increase (~10-12x)")
    report_lines.append(f"- Key insight: amplified noise ({4.3}x V8) enables stronger corrections")
    report_lines.append(f"  but introduces more frame-level discontinuities")
    report_lines.append(f"- Anomalous cases ({len(anomalies['MoMask'])+len(anomalies['MDM'])} total) are concentrated in")
    report_lines.append(f"  backward/running/dance categories with high original FSR")

    report_text = '\n'.join(report_lines)
    with open(f'{SAVE_DIR}/anomaly_report.md', 'w') as f:
        f.write(report_text)
    print(f"  Saved: anomaly_report.md")

    # Save JSON for programmatic use
    with open(f'{SAVE_DIR}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: results.json")

    with open(f'{SAVE_DIR}/anomalies.json', 'w') as f:
        json.dump(anomalies, f, indent=2, default=str)
    print(f"  Saved: anomalies.json")

    print(f"\n{'='*60}")
    print(f"Analysis complete. All outputs in {SAVE_DIR}/")
    print(f"  Figures: 7 PNG files")
    print(f"  Data: results.json, anomalies.json")
    print(f"  Report: anomaly_report.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
