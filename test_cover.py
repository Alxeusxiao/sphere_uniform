"""覆盖目标的回归测试：确认 sphere_data.json 里的 cover 构型真的是
"三个目标里覆盖半径最小"的那一个，且小 N 命中可解析最优。

跑法： python test_cover.py
"""
import json
import math

import sphoptim as so

# 可解析的最优覆盖半径
KNOWN = {
    4: (70.5287793655, '正四面体'),
    6: (54.7356103172, '正八面体'),
    12: (37.3773681406, '正二十面体'),
}

ok = True
data = json.load(open('sphere_data.json', encoding='utf-8'))
keys = sorted(int(k) for k in data)

print("1. 小 N 是否命中可解析最优覆盖半径")
for n, (kn, name) in KNOWN.items():
    if str(n) not in data:
        continue
    P = data[str(n)]['configs']['cover']['pts']
    R = math.degrees(so.coverage_radius(P))
    good = abs(R - kn) < 1e-4
    ok &= good
    print(f"   N={n:>2} {name}: R={R:.5f}  最优={kn:.5f}  差 {R-kn:+.2e}°  {'OK' if good else 'FAIL'}")

print("\n2. cover 构型是否为三个目标里 R_cov 最小者")
bad = []
rows = []
for n in keys:
    c = data[str(n)]['configs']
    Rs = {k: math.degrees(so.coverage_radius(c[k]['pts']))
          for k in ('energy', 'tammes', 'cover')}
    lb = math.degrees(math.acos(max(-1.0, 1 - 2.0 / n)))
    # 容差 1e-7°：不同求解器可能给出同一个构型的浮点等价版本
    is_min = Rs['cover'] <= min(Rs['energy'], Rs['tammes']) + 1e-5   # 1e-5 度：求解器数值噪声
    if not is_min:
        bad.append(n)
    rows.append((n, Rs, lb, is_min))

for n, Rs, lb, is_min in rows:
    tag = 'OK  ' if is_min else 'FAIL'
    dmin = Rs['cover'] - min(Rs['energy'], Rs['tammes'])
    print(f"   {tag} N={n:>4}  R(cover)={Rs['cover']:9.5f}  "
          f"R(energy)={Rs['energy']:9.5f}  R(tammes)={Rs['tammes']:9.5f}  "
          f"下界={lb:8.5f}  cover/下界={Rs['cover']/lb:.4f}  差={dmin:+.1e}")
if bad:
    ok = False
    print(f"\n   不是最小的 N：{bad}")
else:
    print(f"\n   全部 {len(keys)} 组都满足 cover 的 R_cov 最小 ✓")

print("\n3. 与球冠面积下界的关系（R_cov 必须 >= 下界）")
viol = [n for n, Rs, lb, _ in rows if Rs['cover'] < lb - 1e-9]
ok &= not viol
print(f"   违反下界的 N：{viol if viol else '无 ✓'}")

ratios = [Rs['cover'] / lb for _, Rs, lb, _ in rows]
print(f"   cover/下界 比值范围：{min(ratios):.4f} ~ {max(ratios):.4f}")

print("\n" + ("覆盖目标回归测试全部通过 ✓" if ok else "存在 FAIL 项 ✗"))
