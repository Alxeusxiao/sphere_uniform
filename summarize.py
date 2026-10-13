"""从 sphere_data.json 汇总实测质量，生成 README 用的表格。"""
import json
import math

import sphoptim as so

data = json.load(open("sphere_data.json", encoding="utf-8"))
keys = sorted(int(k) for k in data)
print(f"共 {len(keys)} 组：{keys}\n")

hdr = f"{'N':>4} {'能量':>13} {'最优能量':>13} {'偏差':>8} {'dmin(T)':>9} {'最优dmin':>9} {'达成':>7} {'R_cov':>8} {'dmin(C)':>8}"
print(hdr)
print("-" * len(hdr))
rows = []
for n in keys:
    c = data[str(n)]["configs"]
    E = c["energy"]["energy"]
    dT = c["tammes"]["minDeg"]
    R = c["cover"]["radiusDeg"]
    dC = c["cover"]["minDeg"]
    e_ref = ENERGY_REF = {
        4: 3.6742346142, 5: 6.4744091134, 6: 9.9852813742, 7: 14.4522529377,
        8: 19.6752879043, 9: 25.7599865027, 10: 32.7169494604, 11: 40.5964505087,
        12: 49.1652530576, 13: 58.8532306120, 14: 69.3063632971, 15: 80.6702441146,
        16: 92.9116553027, 18: 120.0844674474, 20: 150.8815683387,
    }.get(n)
    d_ref = so.KNOWN_TAMMES_DEG.get(n)
    e_s = f"{e_ref:13.4f}" if e_ref else f"{'-':>13}"
    e_d = f"{100*(E-e_ref)/e_ref:+7.3f}%" if e_ref else f"{'-':>8}"
    d_s = f"{d_ref:9.4f}" if d_ref else f"{'-':>9}"
    d_a = f"{100*dT/d_ref:6.1f}%" if d_ref else f"{'-':>7}"
    print(f"{n:>4} {E:13.4f} {e_s} {e_d} {dT:9.4f} {d_s} {d_a} {R:8.4f} {dC:8.4f}")
    rows.append((n, E, e_ref, dT, d_ref, R))

with_ref = [r for r in rows if r[2]]
print(f"\n能量类：{len(with_ref)} 组有文献最优值，"
      f"最大偏差 {max(abs(100*(r[1]-r[2])/r[2]) for r in with_ref):.4f}%，"
      f"平均偏差 {sum(abs(100*(r[1]-r[2])/r[2]) for r in with_ref)/len(with_ref):.4f}%")
with_d = [r for r in rows if r[4]]
print(f"Tammes：{len(with_d)} 组有最优值，"
      f"最差达成率 {min(100*r[3]/r[4] for r in with_d):.1f}%，"
      f"平均达成率 {sum(100*r[3]/r[4] for r in with_d)/len(with_d):.1f}%")
