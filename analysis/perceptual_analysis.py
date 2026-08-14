"""
Perceptual study analysis.
==========================
Reads all perceptual_*.csv exported by rating.html (default from
outputs/perceptual/responses/) and reports, per comparison:
  - how often the delivery point v19 was preferred vs the alternative (ties shown)
  - a two-sided binomial test on the non-tie votes (H0: 50/50)
  - a per-rater attention check on the FSR=0 control motions

Usage: python analysis/perceptual_analysis.py [responses_dir]
"""
import sys, os, glob, csv
from collections import defaultdict

try:
    from scipy.stats import binomtest
    def pval(k, n): return binomtest(k, n, 0.5).pvalue if n else float('nan')
except Exception:
    import math
    def pval(k, n):  # normal approx fallback
        if not n: return float('nan')
        z = (k - n/2) / (math.sqrt(n)/2)
        return math.erfc(abs(z)/math.sqrt(2))

CONTROLS = {"000076", "000119"}   # FSR=0 motions: conditions ~identical (attention check)


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "outputs/perceptual/responses"
    files = glob.glob(os.path.join(rdir, "perceptual_*.csv"))
    if not files:
        print(f"没有找到打分文件 {rdir}/perceptual_*.csv")
        print("把参与者导出的 CSV 放进该目录后再运行。")
        return

    rows = []
    for fp in files:
        with open(fp) as f:
            rows += list(csv.DictReader(f))
    raters = sorted(set(r['rater'] for r in rows))
    print(f"文件 {len(files)} 份,参与者 {len(raters)}:{', '.join(raters)}")
    print(f"总判断数 {len(rows)}\n")

    # ── 主结果:每类对比 v19 被选比例 ──
    print("="*66)
    print("各对比中「交付点 v19」相对另一方的偏好(排除动作对照集)")
    print("="*66)
    tally = defaultdict(lambda: {'v19': 0, 'other': 0, 'tie': 0, 'other_name': ''})
    for r in rows:
        if r['motion'] in CONTROLS:
            continue
        comp = r['comparison']              # e.g. deskate_vs_v19
        other = comp.replace('_vs_', ' ').split()
        other = [c for c in other if c != 'v19']
        oname = other[0] if other else '?'
        t = tally[comp]; t['other_name'] = oname
        if r['chosen_cond'] == 'v19': t['v19'] += 1
        elif r['chosen_cond'] == 'tie': t['tie'] += 1
        else: t['other'] += 1

    for comp, t in sorted(tally.items()):
        n_dec = t['v19'] + t['other']
        frac = t['v19']/n_dec*100 if n_dec else float('nan')
        p = pval(t['v19'], n_dec)
        star = '  *显著(p<0.05)' if p < 0.05 else '  (不显著)'
        print(f"\n{comp}:")
        print(f"  v19 胜 {t['v19']}   {t['other_name']} 胜 {t['other']}   分不清 {t['tie']}")
        print(f"  非平局中 v19 偏好率 = {frac:.1f}%   binomial p={p:.3g}{star}")

    # ── 注意力检查:对照动作应接近 50/50 或高平局 ──
    print("\n" + "="*66)
    print("注意力检查(FSR=0 对照动作,各条件近乎一致 → 应接近平局/五五开)")
    print("="*66)
    per = defaultdict(lambda: {'dec': 0, 'tie': 0})
    for r in rows:
        if r['motion'] not in CONTROLS:
            continue
        per[r['rater']]['tie' if r['chosen_cond'] == 'tie' else 'dec'] += 1
    for rat, d in sorted(per.items()):
        tot = d['dec'] + d['tie']
        print(f"  {rat}: 对照 {tot} 对中 平局/分不清 {d['tie']}，明确选择 {d['dec']}")

    print("\n说明:若某对比 v19 偏好率显著 >50%,说明该修正(相对另一方)在观感上更自然;"
          "\n若 deskate_vs_v19 不显著,说明学习平滑器相对纯物理无可感知优势(支持主结论)。")


if __name__ == "__main__":
    main()
