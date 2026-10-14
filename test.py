"""全部 Python 自检 / 回归测试（合并自 test_sphoptim + test_cover + check_s_consistency）。

跑法： python test.py
"""
import json
import math
import os
import time

import numpy as np

import sphoptim as so

# ---- 文献最优值 ----
ENERGY_REF = {
    4: 3.6742346142, 5: 6.4744091134, 6: 9.9852813742, 7: 14.4522529377,
    8: 19.6752879043, 9: 25.7599865027, 10: 32.7169494604, 11: 40.5964505087,
    12: 49.1652530576, 13: 58.8532306120, 14: 69.3063632971, 15: 80.6702441146,
    16: 92.9116553027, 18: 120.0844674474, 20: 150.8815683387,
}
# 可解析的最优覆盖半径
COVER_EXACT = {4: (70.5287793655, '正四面体'),
               6: (54.7356103172, '正八面体'),
               12: (37.3773681406, '正二十面体')}

FAILS = []


def chk(name, cond, extra=''):
    print(('OK   ' if cond else 'FAIL ') + name + (('   ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


def emove(p, q, t):
    """沿大圆从 p 朝 q 走弧长 t（严格保持单位长度）。"""
    c = float(np.clip(p @ q, -1, 1))
    sn = math.sqrt(max(1 - c * c, 0.0))
    if sn < 1e-14:
        return p
    ut = (q - c * p) / sn
    return p * math.cos(t) + ut * math.sin(t)


def regular_polyhedra():
    phi = (1 + 5 ** 0.5) / 2
    tetra = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], float)
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    cube = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                     [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]], float)
    ico = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            ico += [[0, s1, s2 * phi], [s1, s2 * phi, 0], [s2 * phi, 0, s1]]
    return tetra, octa, cube, np.array(ico, float)


def part1_geometry():
    print('=' * 84)
    print('1. 球面 Voronoi / 覆盖半径：与正多面体的解析值对照')
    tetra, octa, cube, ico = regular_polyhedra()
    R3 = math.degrees(math.acos(1 / 3))
    cases = [('正四面体', tetra, R3, 109.4712206345, 3),
             ('正八面体', octa, math.degrees(math.acos(1 / math.sqrt(3))), 90.0, 4),
             ('正二十面体', ico, 37.37736814064969, 63.4349488229, 5),
             ('立方体', cube, math.degrees(math.acos(1 / math.sqrt(3))), 70.5287793655, 3)]
    for name, Q, rc_ref, dm_ref, nv in cases:
        Q = so.normalize(Q)
        cells = so.voronoi_cells(Q)
        r = math.degrees(so.coverage_radius(Q))
        dm = math.degrees(so.min_angle(Q))
        rg = math.degrees(so.cover_grid(Q, 300000))
        chk(f'{name} 覆盖半径/夹角/胞腔顶点数',
            abs(r - rc_ref) < 1e-6 and abs(dm - dm_ref) < 1e-6 and all(len(v) == nv for v in cells),
            f'R={r:.5f}(精确 {rc_ref:.5f}) d_min={dm:.5f} 细网格={rg:.4f}')

    # 边界情形：n=1/2/3 不能抛异常
    for n in (2, 3, 4):
        P = so.fibonacci_points(n)
        try:
            r = so.coverage_radius(P)
            cells = so.voronoi_cells(P)
            chk(f'n={n} 边界情形', len(cells) == n, f'R_cov={math.degrees(r):.4f}')
        except Exception as e:
            chk(f'n={n} 边界情形', False, repr(e))


def part2_gradients():
    print('\n2. 梯度：解析式 vs 中心差分（含 s=0 对数能量）')
    rng = np.random.default_rng(11)
    P = so.normalize(rng.normal(size=(6, 3)))
    for s in (1.0, 0.0, 3.0, 6.0, -1.0):
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
        chk(f's={s} 能量-梯度自洽', err < 1e-6, f'最大相对误差 {err:.2e}')

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
            fd = (so.min_angle(A) - so.min_angle(B)) / (2 * eps)
            ut = (A[i] - Q[i]) / eps
            if abs(fd) > 1e-9:
                grad_ok &= abs(fd - float(g[i] @ ut)) / abs(fd) < 1e-4
    chk('d_min 次梯度自洽', grad_ok)


def part3_objectives():
    print('\n3. 四类目标：与已知最优值对照')
    rng = np.random.default_rng(3)
    for n in (4, 6, 8, 12, 20):
        t0 = time.time()
        best_e = min(so.riesz_energy(so.thomson(P0, 1.0, iters=400)[0], 1.0)
                     for P0 in (so.fibonacci_points(n),
                                so.spiral_points(n, jitter=0.2, rng=rng)))
        Pt = so.tammes(so.fibonacci_points(n), iters=6000, rate=0.05, seed=n)[0]
        Pt = so.thomson(Pt, 1.0, iters=400)[0]
        d_t = math.degrees(so.min_angle(Pt))
        res = so.solve_cover(n, seed=n)
        R = res['radius_deg']
        e_ref, d_ref = ENERGY_REF.get(n), so.KNOWN_TAMMES_DEG.get(n)
        msg = (f'E={best_e:.4f}(最优 {e_ref:.4f}) | d_min={d_t:.4f}(最优 '
               f'{d_ref:.4f}) | R_cov={R:.4f} 来源={res["tag"]}')
        ok = (abs(best_e - e_ref) / e_ref < 0.01) and (d_t / d_ref > 0.90)
        chk(f'n={n} 能量与 Tammes', ok, msg + f'  [{time.time()-t0:.1f}s]')


def part_area():
    print('\n4. Voronoi 面积均匀度（第四个目标）')

    # 面积总和必须是 4π（这是 cell_area 最容易写错的地方）
    worst = 0.0
    for n in (4, 6, 12, 20, 50, 100):
        A, _ = so.cell_areas(so.fibonacci_points(n))
        worst = max(worst, abs(float(A.sum()) - 4 * math.pi))
    chk('胞腔面积总和 == 4π', worst < 1e-9, f'最大偏差 {worst:.2e}')

    # 正多面体：胞腔面积全相等 => cv ~ 0
    _, octa, cube, ico = regular_polyhedra()
    for name, Q in (('正二十面体', ico), ('立方体', cube), ('正八面体', octa)):
        Q = so.normalize(Q)
        cv, _ = so.area_uniformity(Q)
        chk(f'{name} 的胞腔面积相等（cv≈0）', cv < 1e-9, f'cv={cv:.2e}')

    # area_lloyd 单调不增
    for n in (12, 30):
        rng = np.random.default_rng(n)
        P0 = so.spiral_points(n, jitter=0.3, rng=rng)
        _, h = so.area_lloyd(P0, iters=60, rate=0.9, power=1.0)
        inc = sum(1 for i in range(len(h) - 1) if h[i + 1] > h[i] + 1e-12)
        chk(f'n={n} area_lloyd 的 cv 单调不增', inc == 0,
            f'{h[0]:.5f} -> {h[-1]:.5f}（上升 {inc} 次）')

    # 数据里 area 构型的 cv 应该最小
    if not os.path.exists('sphere_data.js'):
        print('   （跳过数据检查：没有 sphere_data.js）')
        return
    raw = open('sphere_data.js', encoding='utf-8').read()
    data = json.loads(raw[raw.index('=') + 1: raw.rindex(';')])
    bad, keys = [], sorted(int(k) for k in data if 'area' in data[k]['configs'])
    for n in keys:
        c = data[str(n)]['configs']
        cvs = {k: so.area_uniformity(c[k]['pts'])[0] for k in ('energy', 'tammes', 'cover', 'area')}
        # 容差：不同求解器可能给出"cv 上等价"的构型（N=10 实测各构型只差 5e-7），
        # 所以用相对 + 绝对双容差，避免把浮点噪声判成失败
        best_other = min(cvs['energy'], cvs['tammes'], cvs['cover'])
        if cvs['area'] > best_other + max(1e-5, 1e-3 * best_other):
            bad.append(n)
    chk(f'{len(keys)} 组数据里 area 的 cv 最小', not bad, f'例外 {bad}')
    if keys:
        gains = []
        for n in keys:
            c = data[str(n)]['configs']
            c0 = so.area_uniformity(so.fibonacci_points(n))[0]
            gains.append((c0 - c['area']['cv']) / c0)
        print(f'   相对斐波那契初值的 cv 改善：平均 {100*np.mean(gains):.1f}%，'
              f'最大 {100*np.max(gains):.1f}%')


def part4_cover_regression():
    print('\n5. 覆盖目标回归：数据里 cover 的 R_cov 是否最小、是否命中解析最优')
    if not os.path.exists('sphere_data.js'):
        print('   （跳过：没有 sphere_data.js）')
        return
    raw = open('sphere_data.js', encoding='utf-8').read()
    data = json.loads(raw[raw.index('=') + 1: raw.rindex(';')])
    keys = sorted(int(k) for k in data)

    for n, (kn, name) in COVER_EXACT.items():
        if str(n) not in data:
            continue
        R = math.degrees(so.coverage_radius(data[str(n)]['configs']['cover']['pts']))
        chk(f'N={n} {name} 命中解析最优', abs(R - kn) < 1e-4, f'R={R:.5f} 最优={kn:.5f}')

    bad, lb_viol = [], []
    for n in keys:
        c = data[str(n)]['configs']
        Rs = {k: math.degrees(so.coverage_radius(c[k]['pts'])) for k in ('energy', 'tammes', 'cover')}
        if Rs['cover'] > min(Rs['energy'], Rs['tammes']) + 1e-5:
            bad.append(n)
        lb = math.degrees(math.acos(max(-1.0, 1 - 2.0 / n)))
        if Rs['cover'] < lb - 1e-9:
            lb_viol.append(n)
    chk(f'{len(keys)} 组数据里 cover 的 R_cov 都最小', not bad, f'例外 {bad}')
    chk('R_cov 都不低于球冠面积下界', not lb_viol, f'违反 {lb_viol}')


if __name__ == '__main__':
    part1_geometry()
    part2_gradients()
    part3_objectives()
    part_area()
    part4_cover_regression()
    print('\n' + ('全部通过 ✓' if not FAILS else f'有 {len(FAILS)} 项失败 ✗: {FAILS}'))
