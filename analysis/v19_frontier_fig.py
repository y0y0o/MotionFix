"""
Headline frontier figure: With IK-in-the-loop vs Without vs tuned Gaussian,
three generators, broken y-axis so the without-IK collapse and the real frontier
are both legible. One figure supports contribution #2 AND the central finding.

Data: analysis/v19/ikloop_frontier_full.json (withik/noik, n=200/200/50),
      analysis/v19/transfer.json (Gaussian frontier + uncorrected FSR).
Output: analysis/v19/fig_frontier.png
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8.5,
})
ANA = "analysis/v19"
GENS = [('t2mgpt', 'T2M-GPT'), ('momask', 'MoMask'), ('mdm', 'MDM')]
C = {'withik': '#2a78d6', 'noik': '#c2582b', 'gauss': '#2f8f4e', 'unc': '#6b6b6b'}
C_MUT = '#6b6b6b'


def curve(pts, g):
    j = np.array([p[g]['Jitter'] * 1000 for p in pts])
    f = np.array([p[g]['FSR'] * 100 for p in pts])
    o = np.argsort(j)
    return j[o], f[o]


def main():
    d = json.load(open(f"{ANA}/ikloop_frontier_full.json"))
    tr = json.load(open(f"{ANA}/transfer.json"))['data']

    fig = plt.figure(figsize=(10, 4.4), facecolor='white')
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2.7], hspace=0.08, wspace=0.26)

    for c, (g, title) in enumerate(GENS):
        # data
        wj, wf = curve(d['withik'], g)
        nj, nf = curve(d['noik'], g)
        gfr = tr[g]['gauss_frontier']
        gj = np.array([gfr[s]['Jitter'] * 1000 for s in gfr])
        gf = np.array([gfr[s]['FSR'] * 100 for s in gfr])
        go = np.argsort(gj); gj, gf = gj[go], gf[go]
        unc = tr[g]['original']['FSR'] * 100
        deliv = [p for p in d['withik'] if p['jit_share'] == 0.88][0]

        axT = fig.add_subplot(gs[0, c])
        axB = fig.add_subplot(gs[1, c])

        # bottom: real frontier (withik, gaussian), uncorrected line, delivery point
        for ax in (axT, axB):
            ax.plot(gj, gf, '-s', color=C['gauss'], ms=3, lw=1.3, label='Gaussian', alpha=0.9)
            ax.plot(wj, wf, '-o', color=C['withik'], ms=3.5, lw=1.6, label='With IK-in-the-loop')
            ax.plot(nj, nf, 'o', color=C['noik'], ms=4, label='Without IK-in-the-loop')
            ax.axhline(unc, color=C['unc'], ls='--', lw=1.0, alpha=0.8, label='uncorrected')
        axB.plot(deliv[g]['Jitter'] * 1000, deliv[g]['FSR'] * 100, '*', color=C['withik'],
                 ms=13, mec='black', mew=0.6, zorder=5, label='delivery point (jit-share 0.88)')

        # y limits: top = noik band, bottom = frontier + uncorrected
        lo = min(wf.min(), gf.min()) - 0.4
        hi = max(unc, wf.max(), gf.max()) + 0.4
        axB.set_ylim(lo, hi)
        axT.set_ylim(nf.min() - 0.5, nf.max() + 0.6)

        # common x
        allj = np.concatenate([wj, gj, nj])
        xlo, xhi = allj.min() - 0.4, allj.max() + 0.4
        axT.set_xlim(xlo, xhi); axB.set_xlim(xlo, xhi)

        # broken-axis cosmetics
        axT.spines['bottom'].set_visible(False)
        axB.spines['top'].set_visible(False)
        axT.tick_params(bottom=False, labelbottom=False, colors=C_MUT)
        axB.tick_params(colors=C_MUT)
        for ax in (axT, axB):
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.grid(True, color='#ebeae5', lw=0.7); ax.set_axisbelow(True)
        dk = .012
        for ax, y in ((axT, 0), (axB, 1)):
            kw = dict(transform=ax.transAxes, color='#999', clip_on=False, lw=0.9)
            if y == 0:
                ax.plot((-dk, +dk), (-dk*2, +dk*2), **kw)
            else:
                ax.plot((-dk, +dk), (1-dk, 1+dk), **kw)
        axT.set_title(title, pad=4)
        if c == 0:
            axB.set_ylabel('FSR (%),  lower is better')
        axB.set_xlabel(r'Jitter ($\times10^{-3}$),  lower is better')
        axT.annotate('Without IK: proxy-gamed', xy=(nj.mean(), nf.mean()),
                     xytext=(0, 6), textcoords='offset points', ha='center',
                     fontsize=7.5, color=C['noik'])

    handles, labels = fig.axes[1].get_legend_handles_labels()
    # add delivery marker to legend
    deliv_h = plt.Line2D([], [], marker='*', color=C['withik'], mec='black', mew=0.6,
                         ms=11, ls='none', label='delivery point (jit-share 0.88)')
    order = ['With IK-in-the-loop', 'Gaussian', 'Without IK-in-the-loop', 'uncorrected']
    hmap = dict(zip(labels, handles))
    H = [hmap[k] for k in order if k in hmap] + [deliv_h]
    fig.legend(handles=H, frameon=False, ncol=5, loc='lower center',
               bbox_to_anchor=(0.5, -0.04), labelcolor=C_MUT)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(f"{ANA}/fig_frontier.png", dpi=175, facecolor='white', bbox_inches='tight')
    print(f"-> {ANA}/fig_frontier.png")
    # quick numeric read: withik vs gaussian gap at delivery jitter
    for g, title in GENS:
        wj, wf = curve(d['withik'], g)
        gfr = tr[g]['gauss_frontier']
        gj = np.array([gfr[s]['Jitter']*1000 for s in gfr]); gf = np.array([gfr[s]['FSR']*100 for s in gfr])
        o = np.argsort(gj)
        dv = [p for p in d['withik'] if p['jit_share'] == 0.88][0][g]
        gfsr = np.interp(dv['Jitter']*1000, gj[o], gf[o])
        print(f"{title}: at delivery jitter, WithIK {dv['FSR']*100:.2f}%  Gaussian {gfsr:.2f}%  "
              f"uncorrected {tr[g]['original']['FSR']*100:.2f}%  noik~{curve(d['noik'],g)[1].mean():.1f}%")


if __name__ == "__main__":
    main()
