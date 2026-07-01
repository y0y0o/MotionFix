"""
V18+IK — Results analysis figure (publication-style, 4 panels)
==============================================================
A: FSR  Original vs Learned+IK, per generator (skating reduced)
B: Jitter Original vs Learned+IK, per generator (twitch reduced)
C: FSR–Jitter frontier — Original / Gaussian+IK / Learned+IK per generator
   (shows Learned+IK moves down-left of Original; ties Gaussian on the frontier)
D: Bone-length CV Original vs Learned+IK (unchanged = leg-tear/foot-flip fixed)

Reads analysis/v18_ik_scale/cross_model.json  → analysis/v18_ik_scale/results.png
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ANA = "analysis/v18_ik_scale"
with open(f"{ANA}/cross_model.json") as f:
    C = json.load(f)

gens = ['momask', 'mdm', 't2mgpt']
labels = ['MoMask', 'MDM', 'T2M-GPT']
C_O, C_G, C_L = '#9E9E9E', '#FFB300', '#1E88E5'   # original / gauss / learned

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MotionFix (De-skate → Learned smoother → 2-bone IK): cross-generator results (n=50 each)',
             fontsize=14, fontweight='bold')

x = np.arange(len(gens)); w = 0.35

# ── Panel A: FSR ──
ax = axes[0][0]
fo = [C[g]['original']['FSR']*100 for g in gens]
fl = [C[g]['learn_ik']['FSR']*100 for g in gens]
ax.bar(x-w/2, fo, w, label='Original', color=C_O)
ax.bar(x+w/2, fl, w, label='Learned+IK', color=C_L)
for i,(a,b) in enumerate(zip(fo,fl)):
    ax.text(i-w/2, a+0.15, f'{a:.1f}', ha='center', fontsize=8)
    ax.text(i+w/2, b+0.15, f'{b:.1f}', ha='center', fontsize=8, color=C_L, fontweight='bold')
ax.axhline(5.5, ls='--', color='green', alpha=0.6, label='SOTA (OmniControl ≈5.5%)')
ax.set_title('(A) Foot-Skating Ratio ↓', fontsize=12, fontweight='bold')
ax.set_ylabel('FSR (%)'); ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.25)

# ── Panel B: Jitter ──
ax = axes[0][1]
jo = [C[g]['original']['Jitter'] for g in gens]
jl = [C[g]['learn_ik']['Jitter'] for g in gens]
ax.bar(x-w/2, jo, w, label='Original', color=C_O)
ax.bar(x+w/2, jl, w, label='Learned+IK', color=C_L)
for i,(a,b) in enumerate(zip(jo,jl)):
    ax.text(i-w/2, a+0.0002, f'{a:.4f}', ha='center', fontsize=8)
    ax.text(i+w/2, b+0.0002, f'{b:.4f}', ha='center', fontsize=8, color=C_L, fontweight='bold')
ax.set_title('(B) Jitter (foot accel RMS) ↓', fontsize=12, fontweight='bold')
ax.set_ylabel('Jitter'); ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.25)

# ── Panel C: FSR–Jitter frontier ──
ax = axes[1][0]
for g, lab in zip(gens, labels):
    o, gg, l = C[g]['original'], C[g]['gauss_ik'], C[g]['learn_ik']
    ax.plot([o['FSR']*100, l['FSR']*100], [o['Jitter'], l['Jitter']],
            color='#cccccc', lw=1, zorder=1)
    ax.scatter(o['FSR']*100, o['Jitter'], color=C_O, s=90, zorder=3, edgecolor='k', lw=0.5)
    ax.scatter(gg['FSR']*100, gg['Jitter'], color=C_G, s=90, marker='^', zorder=3, edgecolor='k', lw=0.5)
    ax.scatter(l['FSR']*100, l['Jitter'], color=C_L, s=110, marker='*', zorder=4, edgecolor='k', lw=0.5)
    ax.annotate(lab, (l['FSR']*100, l['Jitter']), fontsize=8,
                textcoords="offset points", xytext=(4,4))
ax.scatter([],[],color=C_O,s=80,edgecolor='k',lw=0.5,label='Original')
ax.scatter([],[],color=C_G,s=80,marker='^',edgecolor='k',lw=0.5,label='Gaussian+IK')
ax.scatter([],[],color=C_L,s=100,marker='*',edgecolor='k',lw=0.5,label='Learned+IK')
ax.set_title('(C) FSR–Jitter frontier (down-left = better)', fontsize=12, fontweight='bold')
ax.set_xlabel('FSR (%)'); ax.set_ylabel('Jitter'); ax.legend(fontsize=8); ax.grid(alpha=0.25)

# ── Panel D: Bone-length CV (leg-tear fixed) ──
ax = axes[1][1]
bo = [C[g]['original']['BoneCV'] for g in gens]
bl = [C[g]['learn_ik']['BoneCV'] for g in gens]
ax.bar(x-w/2, bo, w, label='Original', color=C_O)
ax.bar(x+w/2, bl, w, label='Learned+IK', color=C_L)
for i,(a,b) in enumerate(zip(bo,bl)):
    ax.text(i+w/2, b+0.0005, f'{b:.4f}', ha='center', fontsize=8, color=C_L, fontweight='bold')
ax.set_title('(D) Bone-length CV — unchanged ⇒ no leg-tear / foot-flip', fontsize=12, fontweight='bold')
ax.set_ylabel('BoneCV (lower=rigid)'); ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.25)

fig.tight_layout(rect=[0,0,1,0.97])
out = f"{ANA}/results.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"✓ {out}")
