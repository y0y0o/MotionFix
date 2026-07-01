"""
Visualize physics-based foot skating correction results.
Generates trajectory plots, velocity comparisons, and summary charts.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import compute_fsr, compute_jitter, compute_floating
from utils.physics_fix import physics_foot_fix

INPUT_DIR = "data/test_inputs/momask_50/momask_50_results/no_ik"
OUTPUT_DIR = "outputs/fixed/physics"
VIZ_DIR = "analysis/physics_viz"
os.makedirs(VIZ_DIR, exist_ok=True)

FOOT_JOINTS = [7, 8, 10, 11]
JOINT_NAMES = {7: "L-Ankle", 8: "R-Ankle", 10: "L-Foot", 11: "R-Foot"}
COLORS = {'orig': '#888888', 'fixed': '#2196F3', 'skating': '#FF5252'}

files = sorted(glob.glob(f"{INPUT_DIR}/*.npy"))

# ═══════════════════════════════════════════════════════════════════
# Figure 1: 4-motion trajectory panel (p000021, p001969, p004822, p012529)
# ═══════════════════════════════════════════════════════════════════

showcase = [
    "p000021_rotation_person_is_walking_normally_in_a",
    "p001969_turning_a_man_walks_forward_then_turns",
    "p004822_walking_a_person_is_walking_in_place",
    "p012529_rotation_a_person_is_dancing_the_waltz",
]

fig, axes = plt.subplots(4, 4, figsize=(24, 18))
fig.suptitle('Physics Correction — Foot Trajectories (damp=0.0, freeze)',
             fontsize=16, fontweight='bold', y=0.995)

for row, name in enumerate(showcase):
    fp = f"{INPUT_DIR}/{name}.npy"
    motion = np.load(fp).astype(np.float32)
    if motion.ndim == 4: motion = motion[0]
    fixed, stats = physics_foot_fix(motion, damp_factor=0.0, return_stats=True)

    for col, fj in enumerate(FOOT_JOINTS):
        ax = axes[row][col]
        T = motion.shape[0]
        fsr_b, _, _ = compute_fsr(motion)
        fsr_a, _, _ = compute_fsr(fixed)

        # Plot X trajectory
        ax.plot(motion[:, fj, 0], color='#888888', linewidth=1.0, alpha=0.6, label='Original')
        ax.plot(fixed[:, fj, 0], color='#2196F3', linewidth=1.5, alpha=0.9, label='Physics fix')
        ax.set_title(f'{JOINT_NAMES[fj]} X  |  FSR {fsr_b:.0%}→{fsr_a:.0%}', fontsize=9)
        ax.grid(True, alpha=0.2)
        if row == 0 and col == 0: ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(f'{VIZ_DIR}/01_trajectories_showcase.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 1: 4-motion trajectory showcase")

# ═══════════════════════════════════════════════════════════════════
# Figure 2: FSR before/after bar chart (all 50)
# ═══════════════════════════════════════════════════════════════════

names_short = []
fsr_before = []
fsr_after = []
for fp in files:
    name = os.path.basename(fp).replace('.npy', '')
    names_short.append(name[:35])
    m = np.load(fp).astype(np.float32)
    if m.ndim == 4: m = m[0]
    fsr_b, _, _ = compute_fsr(m)
    fixed, _ = physics_foot_fix(m, damp_factor=0.0, return_stats=True)
    fsr_a, _, _ = compute_fsr(fixed)
    fsr_before.append(fsr_b)
    fsr_after.append(fsr_a)

# Sort by FSR improvement
order = np.argsort([a - b for a, b in zip(fsr_after, fsr_before)])  # most improved first
names_short = [names_short[i] for i in order]
fsr_before = [fsr_before[i] for i in order]
fsr_after = [fsr_after[i] for i in order]

fig, ax = plt.subplots(figsize=(22, 10))
x = np.arange(len(files))
width = 0.35
bars_b = ax.bar(x - width/2, fsr_before, width, color='#888888', alpha=0.7, label=f'Original (avg {np.mean(fsr_before):.1%})')
bars_a = ax.bar(x + width/2, fsr_after, width, color='#2196F3', alpha=0.9, label=f'Physics fix (avg {np.mean(fsr_after):.1%})')
ax.set_xticks(x)
ax.set_xticklabels(names_short, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Foot Skating Ratio')
ax.set_title(f'Figure 2: FSR Before/After — Physics damp=0.0 (avg Δ={np.mean(fsr_after)-np.mean(fsr_before):+.1%})',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.axhline(y=np.mean(fsr_before), color='#888888', linestyle='--', alpha=0.3)
ax.axhline(y=np.mean(fsr_after), color='#2196F3', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(f'{VIZ_DIR}/02_fsr_barchart.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 2: FSR bar chart (50 motions)")

# ═══════════════════════════════════════════════════════════════════
# Figure 3: FSR change histogram
# ═══════════════════════════════════════════════════════════════════
fsr_deltas = [a - b for a, b in zip(fsr_after, fsr_before)]
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(-0.25, 0.15, 30)
ax.hist(fsr_deltas, bins=bins, color='#2196F3', edgecolor='white', alpha=0.85)
improved = sum(1 for d in fsr_deltas if d < 0)
worsened = sum(1 for d in fsr_deltas if d > 0)
unchanged = sum(1 for d in fsr_deltas if d == 0)
ax.axvline(x=0, color='#333333', linestyle='--', linewidth=1.5)
ax.axvline(x=np.mean(fsr_deltas), color='#FF5252', linestyle='-', linewidth=2,
           label=f'Mean Δ={np.mean(fsr_deltas):+.1%}')
ax.set_xlabel('ΔFSR (negative = improved)')
ax.set_ylabel('Count')
ax.set_title(f'Figure 3: FSR Change Distribution — {improved} improved, {worsened} worsened, {unchanged} unchanged',
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(f'{VIZ_DIR}/03_fsr_histogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 3: FSR change histogram")

# ═══════════════════════════════════════════════════════════════════
# Figure 4: FSR vs Jitter scatter
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
jitter_before = []
jitter_after = []
for fp in files:
    m = np.load(fp).astype(np.float32)
    if m.ndim == 4: m = m[0]
    jitter_before.append(compute_jitter(m))
    fixed, _ = physics_foot_fix(m, damp_factor=0.0, return_stats=True)
    jitter_after.append(compute_jitter(fixed))

# Before points
ax.scatter(fsr_before, jitter_before, c='#888888', alpha=0.5, s=40, label='Original')
# After points
ax.scatter(fsr_after, jitter_after, c='#2196F3', alpha=0.7, s=50, label='Physics fix')
# Arrows showing movement
for i in range(len(fsr_before)):
    ax.arrow(fsr_before[i], jitter_before[i],
             fsr_after[i]-fsr_before[i], jitter_after[i]-jitter_before[i],
             head_width=0.002, head_length=0.005, fc='#2196F3', ec='#2196F3', alpha=0.3, width=0.0003)

ax.set_xlabel('FSR', fontsize=12)
ax.set_ylabel('Jitter (m/frame²)', fontsize=12)
ax.set_title('Figure 4: FSR vs Jitter — Before → After Physics Correction', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(f'{VIZ_DIR}/04_fsr_vs_jitter.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 4: FSR vs Jitter scatter")

# ═══════════════════════════════════════════════════════════════════
# Figure 5: Top-6 worst skating motions — trajectory detail
# ═══════════════════════════════════════════════════════════════════
pairs = list(zip(range(len(files)), names_short, fsr_before, fsr_after, fsr_deltas))
# Most improved
most_improved = sorted(pairs, key=lambda x: x[4])[:3]
# Most worsened
most_worsened = sorted(pairs, key=lambda x: -x[4])[:3]
selected = most_improved + most_worsened

fig, axes = plt.subplots(2, 3, figsize=(24, 10))
fig.suptitle('Figure 5: Best vs Worst — Foot Trajectory Detail', fontsize=14, fontweight='bold')

for idx, (orig_idx, name, fsr_b, fsr_a, delta) in enumerate(selected):
    ax = axes[idx // 3][idx % 3]
    fp = files[orig_idx]
    m = np.load(fp).astype(np.float32)
    if m.ndim == 4: m = m[0]
    fixed, _ = physics_foot_fix(m, damp_factor=0.0, return_stats=True)

    # Show L-Ankle X trajectory
    ax.plot(m[:, 7, 0], color='#888888', linewidth=1.0, alpha=0.7, label='Original L-Ankle X')
    ax.plot(fixed[:, 7, 0], color='#2196F3', linewidth=1.8, alpha=0.9, label='Physics fix L-Ankle X')

    color = '#4CAF50' if delta < 0 else '#FF5252'
    ax.set_title(f'{name[:50]}\nFSR {fsr_b:.0%}→{fsr_a:.0%} (Δ={delta:+.1%})',
                 fontsize=9, color=color)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(f'{VIZ_DIR}/05_best_worst_trajectories.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 5: Best/worst trajectory details")

# ═══════════════════════════════════════════════════════════════════
# Figure 6: Multi-metric comparison radar
# ═══════════════════════════════════════════════════════════════════
from utils.metrics import compute_bone_length_consistency, compute_ground_penetration, compute_contact_accuracy, compute_foot_error

metrics_summary = {'FSR_before': [], 'FSR_after': [], 'Jitter_before': [], 'Jitter_after': [],
                   'Floating': [], 'FootErr': [], 'BoneCV': [], 'PenetrationMean': []}

for fp in files:
    m = np.load(fp).astype(np.float32)
    if m.ndim == 4: m = m[0]
    fixed, _ = physics_foot_fix(m, damp_factor=0.0, return_stats=True)

    metrics_summary['FSR_before'].append(compute_fsr(m)[0])
    metrics_summary['FSR_after'].append(compute_fsr(fixed)[0])
    metrics_summary['Jitter_before'].append(compute_jitter(m))
    metrics_summary['Jitter_after'].append(compute_jitter(fixed))
    metrics_summary['Floating'].append(compute_floating(fixed)[0])
    metrics_summary['FootErr'].append(compute_foot_error(fixed, m))
    metrics_summary['BoneCV'].append(compute_bone_length_consistency(fixed))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Figure 6: Per-Motion Metric Distributions', fontsize=14, fontweight='bold')

# FSR distribution comparison
ax = axes[0]
ax.boxplot([metrics_summary['FSR_before'], metrics_summary['FSR_after']],
           labels=['Original', 'Physics fix'], patch_artist=True,
           boxprops=dict(facecolor='#E3F2FD'), medianprops=dict(color='#1565C0'))
ax.set_ylabel('FSR')
ax.set_title('FSR Distribution (n=50)')
ax.grid(True, alpha=0.2, axis='y')

# Jitter distribution comparison
ax = axes[1]
ax.boxplot([metrics_summary['Jitter_before'], metrics_summary['Jitter_after']],
           labels=['Original', 'Physics fix'], patch_artist=True,
           boxprops=dict(facecolor='#FFEBEE'), medianprops=dict(color='#C62828'))
ax.set_ylabel('Jitter (m/frame²)')
ax.set_title('Jitter Distribution (n=50)')
ax.grid(True, alpha=0.2, axis='y')

fig.tight_layout()
fig.savefig(f'{VIZ_DIR}/06_metric_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 6: Metric distribution boxplots")

# ═══════════════════════════════════════════════════════════════════
# Save stats JSON
# ═══════════════════════════════════════════════════════════════════
stats = {
    'n_motions': len(files),
    'FSR': {
        'before_mean': float(np.mean(metrics_summary['FSR_before'])),
        'after_mean': float(np.mean(metrics_summary['FSR_after'])),
        'delta_mean': float(np.mean(metrics_summary['FSR_after']) - np.mean(metrics_summary['FSR_before'])),
        'improved_count': improved,
        'worsened_count': worsened,
    },
    'Jitter': {
        'before_mean': float(np.mean(metrics_summary['Jitter_before'])),
        'after_mean': float(np.mean(metrics_summary['Jitter_after'])),
        'ratio': float(np.mean(metrics_summary['Jitter_after']) / max(np.mean(metrics_summary['Jitter_before']), 1e-8)),
    },
    'FootErr': float(np.mean(metrics_summary['FootErr'])),
    'Floating': float(np.mean(metrics_summary['Floating'])),
    'BoneCV': float(np.mean(metrics_summary['BoneCV'])),
}
with open(f'{VIZ_DIR}/stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n📁 All figures saved to: {VIZ_DIR}/")
print(f"   01_trajectories_showcase.png")
print(f"   02_fsr_barchart.png")
print(f"   03_fsr_histogram.png")
print(f"   04_fsr_vs_jitter.png")
print(f"   05_best_worst_trajectories.png")
print(f"   06_metric_distributions.png")
print(f"   stats.json")
print(f"\n✅ Done.")
