"""自检：精确几何量 / 梯度有限差分 / 三类目标的算法质量。

跑法： python test_sphoptim.py
"""
import math
import time
import numpy as np
import sphoptim as so

np.set_printoptions(precision=6, suppress=True)

# ---- 已知最优值（文献值，用于对照） ----
# Riesz s=1（Thomson 问题）最优能量
ENERGY_REF = {
    4: 3.6742346142, 5: 6.4744091134, 6: 9.9852813742, 7: 14.4522529377,
    8: 19.6752879043, 9: 25.7599865027, 10: 32.7169494604, 11: 40.5964505087,
    12: 49.1652530576, 13: 58.8532306120, 14: 69.3063632971, 15: 80.6702441146,
    16: 92.9116553027, 18: 120.0844674474, 20: 150.8815683387,
}
# Tammes 问题最优最小夹角（度）
DEG_REF = {
    4: 109.4712206345, 5: 90.0, 6: 90.0, 7: 77.8695421459, 8: 74.8584921850,
    9: 70.5287793655, 10: 66.1468220001, 11: 63.4349488229, 12: 63.4349488229,
    13: 57.1367035768, 14: 55.0242404395, 15: 53.6578254161, 16: 52.2443953066,
    18: 49.5561304523, 20: 47.4316362131, 24: 43.6903320963,
    32: 37.3773681406, 60: 30.0,
}


def build_regular():
    phi = (1 + 5 ** 0.5) / 2
    tetra = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], float)
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    ico = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            ico += [[0, s1, s2 * phi], [s1, s2 * phi, 0], [s2 * phi, 0, s1]]
    return tetra, octa, np.array(ico, float)


def emove(p, q, t):
    """沿大圆从 p 朝 q 走弧长 t（严格保持单位长度）。"""
    c = float(np.clip(p @ q, -1, 1))
    sn = math.sqrt(max(1 - c * c, 0.0))
    if sn < 1e-14:
        return p
    ut = (q - c * p) / sn
    return p * math.cos(t) + ut * math.sin(t)


ok = True
print("=" * 84)
print("1. 球面 Voronoi / 覆盖半径：与正多面体的解析值对照")
tetra, octa, ico = build_regular()
cube = so.normalize(np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                              [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]], float))
R3 = math.degrees(math.acos(1 / 3))
cases = [
    # 名称, 点集, 精确覆盖半径, 最小夹角, 每胞腔顶点数
    ("正四面体", tetra, R3, 109.4712206345, 3),
    ("正八面体", octa, math.degrees(math.acos(1 / math.sqrt(3))), 90.0, 4),
    ("正二十面体", ico, 37.37736814064969, 63.4349488229, 5),
    ("立方体", cube, math.degrees(math.acos(1 / math.sqrt(3))), 70.5287793655, 3),
]
for name, Q, rc_ref, dm_ref, nv in cases:
    Q = so.normalize(Q)
    cells = so.voronoi_cells(Q)
    r = math.degrees(so.coverage_radius(Q))
    dm = math.degrees(so.min_angle(Q))
    rg = math.degrees(so.cover_grid(Q, 300000))
    good = abs(r - rc_ref) < 1e-6 and abs(dm - dm_ref) < 1e-6 and all(len(v) == nv for v in cells)
    ok &= good
    print(f"  {name:10s} n={len(Q):2d} 胞腔顶点数={[len(v) for v in cells]}  "
          f"R_cov={r:9.5f}(精确 {rc_ref:9.5f})  d_min={dm:9.5f}  "
          f"[细网格 {rg:8.4f}]  {'OK' if good else 'FAIL'}")

print("\n2. 能量梯度：解析式 vs 中心差分")
rng = np.random.default_rng(11)
P = so.normalize(rng.normal(size=(6, 3)))
for s in (1.0, 3.0):
    G = so.energy_grad(P, s)
    err = 0.0
    for i in range(6):
        q = P[i] + rng.normal(size=3)
        q = q / np.linalg.norm(q)
        eps = 1e-6
        A, B = P.copy(), P.copy()
        A[i] = emove(P[i], q, eps)
        B[i] = emove(P[i], q, -eps)
        fd = (so.riesz_energy(A, s) - so.riesz_energy(B, s)) / (2 * eps)
        ut = (A[i] - P[i]) / eps
        err = max(err, abs(fd - float(G[i] @ ut)) / max(1e-12, abs(fd)))
    print(f"  s={s:4.1f}  最大相对误差={err:.2e}  {'OK' if err < 1e-6 else 'FAIL'}")
    ok &= err < 1e-6

print("\n3. dmin 次梯度：解析式 vs 中心差分")


def dmin_val(Q):
    G = Q @ Q.T
    np.fill_diagonal(G, -2.0)
    return float(np.arccos(np.clip(G.max(), -1, 1)))


grad_ok = True
for _ in range(3):
    Q = so.normalize(rng.normal(size=(7, 3)))
    _, g = so.dmin_grad(Q)
    for i in range(7):
        q = Q[i] + rng.normal(size=3)
        q = q / np.linalg.norm(q)
        eps = 1e-6
        A, B = Q.copy(), Q.copy()
        A[i] = emove(Q[i], q, eps)
        B[i] = emove(Q[i], q, -eps)
        fd = (dmin_val(A) - dmin_val(B)) / (2 * eps)
        ut = (A[i] - Q[i]) / eps
        if abs(fd) > 1e-9:
            grad_ok &= abs(fd - float(g[i] @ ut)) / abs(fd) < 1e-4
print(f"  dmin 次梯度一致性  {'OK' if grad_ok else 'FAIL'}")
ok &= grad_ok

print("\n4. 三类目标的算法质量")
for n in (4, 5, 6, 7, 12, 13, 20, 32):
    t0 = time.time()
    # 能量：多起点
    best_e = None
    for P0 in (so.fibonacci_points(n), so.spiral_points(n, jitter=0.2, rng=rng)):
        Pe, _ = so.thomson(P0, 1.0, iters=400)
        E = so.riesz_energy(Pe, 1.0)
        if best_e is None or E < best_e:
            best_e = E
    # Tammes：次梯度上升 + 能量抛光
    Pt, _ = so.tammes(so.fibonacci_points(n), iters=6000, rate=0.05, seed=n)
    Ptp, _ = so.thomson(Pt, 1.0, iters=400)
    d_t = math.degrees(so.min_angle(Ptp))
    # 覆盖：Lloyd
    Pc, _ = so.coverage_lloyd(so.fibonacci_points(n), iters=200, rate=1.0, decay=0.99)
    R = math.degrees(so.coverage_radius(Pc))
    R0 = math.degrees(so.coverage_radius(so.fibonacci_points(n)))
    e_ref = ENERGY_REF.get(n)
    d_ref = DEG_REF.get(n)
    e_txt = f"{e_ref:10.4f} ({100*(best_e-e_ref)/e_ref:+.3f}%)" if e_ref else "     -     "
    d_txt = f"{d_ref:8.4f} ({100*d_t/d_ref:5.1f}%)" if d_ref else "    -    "
    print(f"  n={n:3d} {time.time()-t0:5.1f}s | E={best_e:10.4f} vs 最优 {e_txt} "
          f"| dmin={d_t:8.4f} vs 最优 {d_txt} | R_cov={R:7.3f} (初始 {R0:7.3f})")
    if e_ref:
        ok &= abs(best_e - e_ref) / e_ref < 0.01
    if d_ref and n not in (32,):
        ok &= d_t / d_ref > 0.90
    ok &= R < R0 + 1e-9

print("\n5. 退化与边界情形")
edge_ok = True
for n in (2, 3, 4):
    P = so.fibonacci_points(n)
    try:
        r = so.coverage_radius(P)
        d = math.degrees(so.min_angle(P))
        cells = so.voronoi_cells(P)
        good = len(cells) == n
        edge_ok &= good
        print(f"  n={n} R_cov={math.degrees(r):8.4f} dmin={d:8.4f} "
              f"胞腔顶点数={[len(v) for v in cells]} {'OK' if good else 'FAIL'}")
    except Exception as e:
        edge_ok = False
        print(f"  n={n} 异常: {e!r}  FAIL")
ok &= edge_ok

print("\n" + ("全部通过" if ok else "存在 FAIL 项，见上"))
