# V13 Anomaly Analysis Report

## Summary Statistics

### MoMask
- Motions analyzed: 50
- FSR: 14.1% → 10.3% (-3.8%)
- Jitter: 0.0128 → 0.1580 (12.3x)
- Mean foot error: 7.6 cm
- Improved: 39, Worsened: 6, Unchanged: 5

**Best FSR improvements:**
  - p004822_walking_a_person_is_walking_in_p: 40.3%→21.3% (-19.0%), jitter 5x, err 3.8cm
  - p007561_complex_a_man_walks_in_a_curve: 21.9%→7.0% (-14.9%), jitter 23x, err 16.9cm
  - p009161_walking_a_person_walks_forward_f: 21.4%→8.3% (-13.1%), jitter 14x, err 4.8cm

**Worst FSR regressions:**
  - p002627_dance_a_person_dancing_starting_: 39.7%→45.5% (+5.7%), jitter 6x, err 6.1cm
  - p000818_rotation_a_person_is_spinning_in: 48.6%→51.3% (+2.7%), jitter 3x, err 5.0cm
  - p003424_jumping_man_jumps_twice_in_place: 0.8%→1.6% (+0.8%), jitter 1x, err 0.1cm

### MDM
- Motions analyzed: 50
- FSR: 11.9% → 7.8% (-4.1%)
- Jitter: 0.0142 → 0.1161 (8.2x)
- Mean foot error: 4.0 cm
- Improved: 33, Worsened: 10, Unchanged: 7

**Best FSR improvements:**
  - mdm_004822: 88.9%→39.2% (-49.7%), jitter 3x, err 6.5cm
  - mdm_001168: 16.1%→3.2% (-12.9%), jitter 20x, err 7.2cm
  - mdm_010665: 12.6%→1.1% (-11.5%), jitter 13x, err 4.1cm

**Worst FSR regressions:**
  - mdm_003424: 4.8%→9.5% (+4.8%), jitter 2x, err 0.3cm
  - mdm_011028: 11.1%→14.9% (+3.8%), jitter 4x, err 1.5cm
  - mdm_011684: 2.9%→5.2% (+2.3%), jitter 2x, err 0.2cm

## Category Breakdown

### MoMask
  complex     : n= 3, FSR change= -6.3pp, jitter=6x, err=5cm
  dance       : n=11, FSR change= -1.5pp, jitter=8x, err=4cm
  jumping     : n= 7, FSR change= -1.7pp, jitter=4x, err=2cm
  running     : n= 2, FSR change= -2.3pp, jitter=26x, err=28cm
  turning     : n= 3, FSR change= -1.0pp, jitter=6x, err=4cm
  walking     : n=24, FSR change= -5.6pp, jitter=16x, err=10cm

### MDM
  other       : n=50, FSR change= -4.1pp, jitter=8x, err=4cm

## Anomaly Details

### MoMask — 42 anomalies

**p000021** (walking)
  - FSR: 22.5% → 16.7% (-5.8%)
  - Jitter: 0.0110 → 0.2883 (26x)
  - Foot error: 19.9cm avg, 178.7cm max
  - Issues: Max displacement: 1.787m

**p000818** (turning)
  - FSR: 48.6% → 51.3% (+2.7%)
  - Jitter: 0.0452 → 0.1186 (3x)
  - Foot error: 5.0cm avg, 72.4cm max
  - Issues: Max displacement: 0.724m

**p001120** (walking)
  - FSR: 9.9% → 4.0% (-5.9%)
  - Jitter: 0.0094 → 0.1213 (13x)
  - Foot error: 2.6cm avg, 93.9cm max
  - Issues: Max displacement: 0.939m

**p001168** (walking)
  - FSR: 4.0% → 1.7% (-2.3%)
  - Jitter: 0.0074 → 0.0677 (9x)
  - Foot error: 0.9cm avg, 96.2cm max
  - Issues: Max displacement: 0.962m

**p001448** (turning)
  - FSR: 12.5% → 7.6% (-4.9%)
  - Jitter: 0.0139 → 0.1006 (7x)
  - Foot error: 2.8cm avg, 69.5cm max
  - Issues: Max displacement: 0.695m

**p001567** (dance)
  - FSR: 5.2% → 5.2% (-0.0%)
  - Jitter: 0.0061 → 0.0801 (13x)
  - Foot error: 2.2cm avg, 84.8cm max
  - Issues: Max displacement: 0.848m

**p001969** (walking)
  - FSR: 19.2% → 15.6% (-3.6%)
  - Jitter: 0.0115 → 0.0983 (9x)
  - Foot error: 4.6cm avg, 75.7cm max
  - Issues: Max displacement: 0.757m

**p002104** (walking)
  - FSR: 28.4% → 19.1% (-9.3%)
  - Jitter: 0.0143 → 0.3618 (25x)
  - Foot error: 17.2cm avg, 215.5cm max
  - Issues: Max displacement: 2.155m

**p002530** (walking)
  - FSR: 14.0% → 12.2% (-1.8%)
  - Jitter: 0.0078 → 0.1154 (15x)
  - Foot error: 5.2cm avg, 84.5cm max
  - Issues: Max displacement: 0.845m

**p002627** (dance)
  - FSR: 39.7% → 45.5% (+5.7%)
  - Jitter: 0.0175 → 0.0990 (6x)
  - Foot error: 6.1cm avg, 55.2cm max
  - Issues: FSR worsened: +5.7%, Max displacement: 0.552m

**p003005** (dance)
  - FSR: 6.5% → 4.4% (-2.1%)
  - Jitter: 0.0087 → 0.0719 (8x)
  - Foot error: 2.6cm avg, 72.1cm max
  - Issues: Max displacement: 0.721m

**p003111** (dance)
  - FSR: 7.0% → 2.0% (-5.1%)
  - Jitter: 0.0245 → 0.3973 (16x)
  - Foot error: 8.8cm avg, 213.6cm max
  - Issues: Max displacement: 2.136m

**p003566** (walking)
  - FSR: 21.3% → 15.5% (-5.9%)
  - Jitter: 0.0157 → 0.4854 (31x)
  - Foot error: 21.6cm avg, 260.6cm max
  - Issues: Jitter explosion: 31x, Large foot error: 0.216m, Max displacement: 2.606m

**p003784** (walking)
  - FSR: 12.4% → 11.2% (-1.2%)
  - Jitter: 0.0108 → 0.1503 (14x)
  - Foot error: 5.1cm avg, 118.0cm max
  - Issues: Max displacement: 1.180m

**p004040** (dance)
  - FSR: 3.3% → 2.6% (-0.7%)
  - Jitter: 0.0225 → 0.0805 (4x)
  - Foot error: 1.3cm avg, 76.8cm max
  - Issues: Max displacement: 0.768m

**p004311** (complex)
  - FSR: 21.3% → 12.9% (-8.4%)
  - Jitter: 0.0114 → 0.1140 (10x)
  - Foot error: 6.8cm avg, 78.0cm max
  - Issues: Max displacement: 0.780m

**p004488** (walking)
  - FSR: 12.2% → 9.8% (-2.4%)
  - Jitter: 0.0097 → 0.1101 (11x)
  - Foot error: 4.8cm avg, 102.3cm max
  - Issues: Max displacement: 1.023m

**p004822** (walking)
  - FSR: 40.3% → 21.3% (-19.0%)
  - Jitter: 0.0104 → 0.0511 (5x)
  - Foot error: 3.8cm avg, 30.0cm max
  - Issues: Large FSR fix: 40.3%→21.3%

**p004965** (walking)
  - FSR: 2.7% → 1.1% (-1.6%)
  - Jitter: 0.0054 → 0.0372 (7x)
  - Foot error: 1.0cm avg, 50.8cm max
  - Issues: Max displacement: 0.508m

**p006658** (walking)
  - FSR: 11.1% → 9.5% (-1.6%)
  - Jitter: 0.0082 → 0.1353 (17x)
  - Foot error: 6.1cm avg, 114.0cm max
  - Issues: Max displacement: 1.140m

**p006701** (walking)
  - FSR: 12.1% → 12.2% (+0.1%)
  - Jitter: 0.0075 → 0.0631 (8x)
  - Foot error: 2.3cm avg, 52.8cm max
  - Issues: Max displacement: 0.528m

**p007561** (walking)
  - FSR: 21.9% → 7.0% (-14.9%)
  - Jitter: 0.0144 → 0.3390 (23x)
  - Foot error: 16.9cm avg, 196.5cm max
  - Issues: Max displacement: 1.965m

**p007767** (dance)
  - FSR: 18.8% → 9.0% (-9.8%)
  - Jitter: 0.0198 → 0.2351 (12x)
  - Foot error: 9.9cm avg, 146.1cm max
  - Issues: Max displacement: 1.461m

**p008292** (turning)
  - FSR: 12.5% → 11.7% (-0.8%)
  - Jitter: 0.0111 → 0.0930 (8x)
  - Foot error: 4.1cm avg, 82.7cm max
  - Issues: Max displacement: 0.827m

**p008463** (walking)
  - FSR: 12.3% → 9.5% (-2.7%)
  - Jitter: 0.0111 → 0.1258 (11x)
  - Foot error: 5.1cm avg, 83.1cm max
  - Issues: Max displacement: 0.831m

**p009161** (walking)
  - FSR: 21.4% → 8.3% (-13.1%)
  - Jitter: 0.0127 → 0.1761 (14x)
  - Foot error: 4.8cm avg, 156.5cm max
  - Issues: Max displacement: 1.565m

**p009377** (walking)
  - FSR: 15.8% → 7.3% (-8.4%)
  - Jitter: 0.0090 → 0.2967 (33x)
  - Foot error: 12.2cm avg, 224.1cm max
  - Issues: Jitter explosion: 33x, Max displacement: 2.241m

**p009539** (walking)
  - FSR: 5.1% → 2.7% (-2.4%)
  - Jitter: 0.0085 → 0.0632 (7x)
  - Foot error: 1.1cm avg, 86.5cm max
  - Issues: Max displacement: 0.865m

**p009613** (running)
  - FSR: 28.8% → 27.4% (-1.4%)
  - Jitter: 0.0280 → 1.0305 (37x)
  - Foot error: 47.6cm avg, 479.1cm max
  - Issues: Jitter explosion: 37x, Large foot error: 0.476m, Max displacement: 4.791m

**p009958** (walking)
  - FSR: 52.1% → 43.0% (-9.1%)
  - Jitter: 0.0115 → 0.6462 (56x)
  - Foot error: 83.2cm avg, 249.0cm max
  - Issues: Large FSR fix: 52.1%→43.0%, Jitter explosion: 56x, Large foot error: 0.832m, Max displacement: 2.490m

**p010665** (walking)
  - FSR: 9.5% → 9.5% (+0.1%)
  - Jitter: 0.0081 → 0.0728 (9x)
  - Foot error: 2.3cm avg, 60.1cm max
  - Issues: Max displacement: 0.601m

**p010819** (walking)
  - FSR: 16.9% → 15.0% (-1.9%)
  - Jitter: 0.0130 → 0.1613 (12x)
  - Foot error: 8.3cm avg, 111.6cm max
  - Issues: Max displacement: 1.116m

**p011028** (jumping)
  - FSR: 14.0% → 5.5% (-8.4%)
  - Jitter: 0.0241 → 0.1599 (7x)
  - Foot error: 6.0cm avg, 113.4cm max
  - Issues: Max displacement: 1.134m

**p011441** (walking)
  - FSR: 19.2% → 18.8% (-0.5%)
  - Jitter: 0.0125 → 0.0870 (7x)
  - Foot error: 3.4cm avg, 52.0cm max
  - Issues: Max displacement: 0.520m

**p011673** (jumping)
  - FSR: 1.4% → 0.0% (-1.4%)
  - Jitter: 0.0065 → 0.0447 (7x)
  - Foot error: 0.9cm avg, 80.1cm max
  - Issues: Max displacement: 0.801m

**p011978** (walking)
  - FSR: 20.3% → 9.1% (-11.2%)
  - Jitter: 0.0147 → 0.2522 (17x)
  - Foot error: 13.4cm avg, 203.4cm max
  - Issues: Max displacement: 2.034m

**p011997** (walking)
  - FSR: 12.7% → 3.1% (-9.6%)
  - Jitter: 0.0086 → 0.0338 (4x)
  - Foot error: 0.6cm avg, 55.5cm max
  - Issues: Max displacement: 0.555m

**p012309** (jumping)
  - FSR: 10.6% → 7.2% (-3.5%)
  - Jitter: 0.0170 → 0.1356 (8x)
  - Foot error: 4.5cm avg, 88.8cm max
  - Issues: Max displacement: 0.888m

**p012498** (complex)
  - FSR: 17.6% → 7.2% (-10.4%)
  - Jitter: 0.0200 → 0.1243 (6x)
  - Foot error: 6.6cm avg, 82.2cm max
  - Issues: Max displacement: 0.822m

**p012529** (dance)
  - FSR: 19.3% → 16.9% (-2.4%)
  - Jitter: 0.0097 → 0.0937 (10x)
  - Foot error: 6.5cm avg, 73.3cm max
  - Issues: Max displacement: 0.733m

**p012657** (dance)
  - FSR: 2.0% → 0.5% (-1.5%)
  - Jitter: 0.0202 → 0.2017 (10x)
  - Foot error: 1.8cm avg, 141.3cm max
  - Issues: Max displacement: 1.413m

**p013207** (running)
  - FSR: 6.9% → 3.6% (-3.2%)
  - Jitter: 0.0180 → 0.2601 (14x)
  - Foot error: 7.8cm avg, 200.7cm max
  - Issues: Max displacement: 2.007m

### MDM — 32 anomalies

**p** (other)
  - FSR: 26.2% → 20.4% (-5.8%)
  - Jitter: 0.0202 → 0.3213 (16x)
  - Foot error: 12.8cm avg, 188.0cm max
  - Issues: Max displacement: 1.880m

**p** (other)
  - FSR: 13.2% → 11.1% (-2.1%)
  - Jitter: 0.0199 → 0.1117 (6x)
  - Foot error: 4.6cm avg, 90.5cm max
  - Issues: Max displacement: 0.905m

**p** (other)
  - FSR: 15.0% → 6.4% (-8.7%)
  - Jitter: 0.0075 → 0.1320 (18x)
  - Foot error: 6.7cm avg, 113.5cm max
  - Issues: Max displacement: 1.135m

**p** (other)
  - FSR: 16.1% → 3.2% (-12.9%)
  - Jitter: 0.0089 → 0.1825 (20x)
  - Foot error: 7.2cm avg, 148.1cm max
  - Issues: Max displacement: 1.481m

**p** (other)
  - FSR: 12.8% → 9.7% (-3.0%)
  - Jitter: 0.0112 → 0.1163 (10x)
  - Foot error: 5.6cm avg, 93.6cm max
  - Issues: Max displacement: 0.936m

**p** (other)
  - FSR: 18.2% → 15.0% (-3.3%)
  - Jitter: 0.0174 → 0.1323 (8x)
  - Foot error: 4.9cm avg, 91.6cm max
  - Issues: Max displacement: 0.916m

**p** (other)
  - FSR: 21.2% → 11.2% (-10.1%)
  - Jitter: 0.0202 → 0.2522 (13x)
  - Foot error: 10.5cm avg, 166.5cm max
  - Issues: Max displacement: 1.665m

**p** (other)
  - FSR: 11.3% → 6.3% (-5.0%)
  - Jitter: 0.0084 → 0.1002 (12x)
  - Foot error: 3.7cm avg, 95.9cm max
  - Issues: Max displacement: 0.959m

**p** (other)
  - FSR: 22.2% → 18.1% (-4.1%)
  - Jitter: 0.0432 → 0.1332 (3x)
  - Foot error: 4.7cm avg, 57.9cm max
  - Issues: Max displacement: 0.579m

**p** (other)
  - FSR: 19.4% → 16.0% (-3.4%)
  - Jitter: 0.0253 → 0.2461 (10x)
  - Foot error: 8.7cm avg, 125.6cm max
  - Issues: Max displacement: 1.256m

**p** (other)
  - FSR: 22.2% → 16.6% (-5.6%)
  - Jitter: 0.0156 → 0.1748 (11x)
  - Foot error: 8.4cm avg, 103.9cm max
  - Issues: Max displacement: 1.039m

**p** (other)
  - FSR: 13.4% → 11.4% (-2.0%)
  - Jitter: 0.0093 → 0.0887 (9x)
  - Foot error: 5.0cm avg, 57.0cm max
  - Issues: Max displacement: 0.570m

**p** (other)
  - FSR: 11.3% → 7.6% (-3.7%)
  - Jitter: 0.0102 → 0.1273 (12x)
  - Foot error: 4.5cm avg, 103.9cm max
  - Issues: Max displacement: 1.039m

**p** (other)
  - FSR: 88.9% → 39.2% (-49.7%)
  - Jitter: 0.0183 → 0.0594 (3x)
  - Foot error: 6.5cm avg, 22.1cm max
  - Issues: Large FSR fix: 88.9%→39.2%

**p** (other)
  - FSR: 18.8% → 12.4% (-6.4%)
  - Jitter: 0.0191 → 0.1938 (10x)
  - Foot error: 9.3cm avg, 107.3cm max
  - Issues: Max displacement: 1.073m

**p** (other)
  - FSR: 7.0% → 4.5% (-2.6%)
  - Jitter: 0.0171 → 0.3181 (19x)
  - Foot error: 10.5cm avg, 203.6cm max
  - Issues: Max displacement: 2.036m

**p** (other)
  - FSR: 10.8% → 5.3% (-5.5%)
  - Jitter: 0.0102 → 0.1027 (10x)
  - Foot error: 3.7cm avg, 96.4cm max
  - Issues: Max displacement: 0.964m

**p** (other)
  - FSR: 16.0% → 6.2% (-9.9%)
  - Jitter: 0.0159 → 0.1378 (9x)
  - Foot error: 5.8cm avg, 110.0cm max
  - Issues: Max displacement: 1.100m

**p** (other)
  - FSR: 14.8% → 10.3% (-4.5%)
  - Jitter: 0.0131 → 0.1514 (12x)
  - Foot error: 6.3cm avg, 101.9cm max
  - Issues: Max displacement: 1.019m

**p** (other)
  - FSR: 17.4% → 11.0% (-6.4%)
  - Jitter: 0.0191 → 0.1382 (7x)
  - Foot error: 5.4cm avg, 83.9cm max
  - Issues: Max displacement: 0.839m

**p** (other)
  - FSR: 12.1% → 6.9% (-5.1%)
  - Jitter: 0.0105 → 0.1610 (15x)
  - Foot error: 4.6cm avg, 122.7cm max
  - Issues: Max displacement: 1.227m

**p** (other)
  - FSR: 14.7% → 10.6% (-4.1%)
  - Jitter: 0.0160 → 0.1395 (9x)
  - Foot error: 4.6cm avg, 87.6cm max
  - Issues: Max displacement: 0.876m

**p** (other)
  - FSR: 8.8% → 1.8% (-7.0%)
  - Jitter: 0.0069 → 0.1339 (19x)
  - Foot error: 3.9cm avg, 102.8cm max
  - Issues: Max displacement: 1.028m

**p** (other)
  - FSR: 13.3% → 11.4% (-2.0%)
  - Jitter: 0.0146 → 0.1462 (10x)
  - Foot error: 1.6cm avg, 163.7cm max
  - Issues: Max displacement: 1.637m

**p** (other)
  - FSR: 13.5% → 11.3% (-2.1%)
  - Jitter: 0.0143 → 0.2261 (16x)
  - Foot error: 10.6cm avg, 107.1cm max
  - Issues: Max displacement: 1.071m

**p** (other)
  - FSR: 12.6% → 1.1% (-11.5%)
  - Jitter: 0.0075 → 0.0980 (13x)
  - Foot error: 4.1cm avg, 94.9cm max
  - Issues: Max displacement: 0.949m

**p** (other)
  - FSR: 14.1% → 11.5% (-2.6%)
  - Jitter: 0.0118 → 0.0697 (6x)
  - Foot error: 3.6cm avg, 64.3cm max
  - Issues: Max displacement: 0.643m

**p** (other)
  - FSR: 11.1% → 14.9% (+3.8%)
  - Jitter: 0.0189 → 0.0741 (4x)
  - Foot error: 1.5cm avg, 61.0cm max
  - Issues: Max displacement: 0.610m

**p** (other)
  - FSR: 11.8% → 5.9% (-5.8%)
  - Jitter: 0.0136 → 0.1616 (12x)
  - Foot error: 5.8cm avg, 129.8cm max
  - Issues: Max displacement: 1.298m

**p** (other)
  - FSR: 4.6% → 3.7% (-0.9%)
  - Jitter: 0.0155 → 0.1384 (9x)
  - Foot error: 2.3cm avg, 101.1cm max
  - Issues: Max displacement: 1.011m

**p** (other)
  - FSR: 5.1% → 0.0% (-5.1%)
  - Jitter: 0.0172 → 0.2087 (12x)
  - Foot error: 3.8cm avg, 154.1cm max
  - Issues: Max displacement: 1.541m

**p** (other)
  - FSR: 5.2% → 0.7% (-4.5%)
  - Jitter: 0.0264 → 0.5617 (21x)
  - Foot error: 8.6cm avg, 330.5cm max
  - Issues: Max displacement: 3.305m

## Overall Assessment

- V13 is the best-performing version for FSR reduction
  - MoMask: -3.8pp (V8: -2.9pp)
  - MDM: -4.1pp (V8: -4.0pp)
- Main weakness: jitter increase (~10-12x)
- Key insight: amplified noise (4.3x V8) enables stronger corrections
  but introduces more frame-level discontinuities
- Anomalous cases (74 total) are concentrated in
  backward/running/dance categories with high original FSR