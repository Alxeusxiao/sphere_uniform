"""球面 k-center（最小化覆盖半径 R_cov）局部搜索。

目标：R_cov(P) = max_x min_i angle(x, p_i)

做法：对每个 site，取自己 Voronoi 胞腔内"并列最远"的顶点/棱中点的**平均方向**
作为移动方向，朝它走一步；每一轮扫描后用**精确 R_cov 做整轮回退**——只要这一轮
没有让 R_cov 下降，就整轮回滚并把步长减半。这样 R_cov 单调不增。

为什么取"并列最远点的平均"而不是"单个最远点"：
  正八面体（N=6）、正二十面体（N=12）本身已是局部最优，它们的每个胞腔有 3 个
  并列最远顶点；朝单个顶点移动会破坏对称性（实测 R_cov 从 54.7° 恶化到 73°），
  而朝平均方向移动时三个方向互相抵消 → 平均方向为零 → 判为局部最优、停住。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from sphoptim import (coverage_radius, normalize, voronoi_cells, _arc_dir,
                      cells_and_radius)


def cell_worst_set(P: np.ndarray, i: int, V: np.ndarray,
                   tol: float = 0.02) -> Optional[np.ndarray]:
    """site i 胞腔内"并列最远"的顶点/棱中点集合（(m,3) 单位向量）。"""
    if len(V) < 3:
        return np.array([-P[i]])                      # 退化胞腔（半球）：对跖点
    cand = list(V)
    W = np.roll(V, -1, axis=0)
    mids = V + W
    nm = np.linalg.norm(mids, axis=1, keepdims=True)
    mids = mids / np.where(nm > 1e-15, nm, 1.0)
    cand.extend(list(mids))
    C = np.array(cand)
    d = np.arccos(np.clip(C @ P[i], -1.0, 1.0))
    dmax = float(d.max())
    sel = C[d >= dmax * (1.0 - tol)]
    return sel if len(sel) else None


def worst_avg_direction(P: np.ndarray, i: int, V: np.ndarray,
                        tol: float) -> Optional[np.ndarray]:
    """site i 的移动方向：并列最远方向的平均单位向量；无改进方向时返回 None。"""
    sel = cell_worst_set(P, i, V, tol)
    if sel is None or len(sel) == 0:
        return None
    avg = sel.sum(axis=0)
    na = float(np.linalg.norm(avg))
    if na < 1e-12:
        return None                                   # 方向相消 => 已局部最优
    return avg / na


def kcenter_scan(Q: np.ndarray, step: float, tol: float) -> Tuple[np.ndarray, int, float]:
    """一轮扫描：逐点朝自己的最远方向走一小步。

    返回 (新点集, 移动点数, 新构型的精确 R_cov)——
    R_cov 顺带在同一次胞腔遍历里算出来，避免外层重复构造 Voronoi。
    """
    cells = voronoi_cells(Q)
    moved = 0
    for i in range(len(Q)):
        w = worst_avg_direction(Q, i, cells[i], tol)
        if w is None:
            continue
        ut = _arc_dir(Q[i], w)
        if ut is None:
            continue
        Q[i] = Q[i] * math.cos(step) + ut * math.sin(step)
        Q[i] /= np.linalg.norm(Q[i])
        moved += 1
    _, R = cells_and_radius(Q)
    return Q, moved, R


def kcenter_local_search(P: np.ndarray, iters: int = 150, step: float = 0.25,
                         tols: Tuple[float, ...] = (0.02, 0.08, 0.2),
                         callback=None, callback_every: int = 10,
                         history_every: int = 1) -> Tuple[np.ndarray, List[float]]:
    """整轮回退的 k-center 局部搜索（R_cov 单调不增）。"""
    Q = normalize(np.array(P, dtype=float, copy=True))
    _, R = cells_and_radius(Q)
    hist: List[float] = [R]
    a = step
    for it in range(iters):
        improved = False
        for tol in tols:                              # 依次尝试不同的"并列"容差
            if a < 1e-7:
                break
            trial, moved, R_try = kcenter_scan(Q.copy(), a, tol)
            if moved == 0:
                continue
            if R_try < R - 1e-14:
                Q, R = trial, R_try
                improved = True
                a = min(step, a * 1.2)
                break
        if not improved:
            a *= 0.5
            if a < 1e-6:
                break
        if it % history_every == 0:
            hist.append(R)
        if callback is not None and (it + 1) % callback_every == 0:
            callback(it + 1, Q)
    _, Rend = cells_and_radius(Q)
    hist.append(Rend)
    return Q, hist


def solve_kcenter(n: int, seed: int = 0, iters: int = 150,
                  starts: Optional[List[np.ndarray]] = None,
                  polish_with_lloyd: bool = True) -> Dict:
    """多起点求解覆盖最小构型（可选再用球面 Lloyd 抛光）。"""
    from sphoptim import fibonacci_points, spiral_points, random_points, coverage_lloyd
    rng = np.random.default_rng(seed)
    if starts is None:
        starts = [fibonacci_points(n),
                  spiral_points(n, jitter=0.2, rng=rng),
                  spiral_points(n, jitter=0.45, rng=rng),
                  random_points(n, rng)]
    best = None
    for P0 in starts:
        Q, h = kcenter_local_search(P0, iters=iters)
        if polish_with_lloyd:
            # Lloyd 不保证改善覆盖，只在确实更优时采纳（多给一个局部最优候选）
            Ql, _ = coverage_lloyd(Q, iters=40, rate=1.0, decay=0.99)
            if coverage_radius(Ql) < coverage_radius(Q):
                Q = Ql
        R = coverage_radius(Q)
        if best is None or R < best[0]:
            best = (R, Q, h)
    return {"points": best[1], "radius_deg": math.degrees(best[0]), "history": best[2]}
