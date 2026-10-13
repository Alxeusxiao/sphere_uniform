"""给 k-center 局部搜索加"逃逸候选，突破局部极小。

卡在局部极小时，依次尝试这些候选构型，只要精确 R_cov 下降就采纳：
  A) 逐点朝"单个最远方向"移动（打破对称的小扰动，步长很小）
  B) 球面 Lloyd 一步（朝胞腔质心；它单调降低均方距离，常能把构型带出鞍点）
  C) 能量（Thomson s=1）精修一步（能量最优构型往往也接近覆盖最优）

所有候选都用精确 R_cov 判定，因此整体仍然单调不增。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

import sphoptim as so
from kcenter import cell_worst_set, kcenter_scan, kcenter_local_search, worst_avg_direction


def _lloyd_one(Q: np.ndarray) -> np.ndarray:
    """球面 Lloyd 一步（朝各胞腔面积加权质心方向移动）。"""
    cells = so.voronoi_cells(Q)
    out = Q.copy()
    for i, V in enumerate(cells):
        c = so.cell_centroid_dirs(V)
        if c is None:
            continue
        t = math.acos(float(np.clip(c @ Q[i], -1.0, 1.0)))
        if t < 1e-12:
            continue
        tan = (c - Q[i] * float(c @ Q[i])) / math.sin(t)
        out[i] = Q[i] * math.cos(t) + tan * math.sin(t)
        out[i] /= np.linalg.norm(out[i])
    return out


def _single_worst_step(Q: np.ndarray, step: float, tol: float) -> Optional[np.ndarray]:
    """朝每个点的"单个最远方向"各走一小步（对称破缺用的小扰动）。"""
    cells = so.voronoi_cells(Q)
    R = so.coverage_radius(Q)
    best = None
    for i, V in enumerate(cells):
        sel = cell_worst_set(Q, i, V, tol)
        if sel is None or len(sel) == 0:
            continue
        for w in sel:
            ut = so._arc_dir(Q[i], w)
            if ut is None:
                continue
            trial = Q.copy()
            trial[i] = Q[i] * math.cos(step) + ut * math.sin(step)
            trial[i] /= np.linalg.norm(trial[i])
            Rt = so.coverage_radius(trial)
            if Rt < R - 1e-14 and (best is None or Rt < best[0]):
                best = (Rt, trial)
    return None if best is None else best[1]


def _energy_polish_step(Q: np.ndarray, s: float = 1.0) -> Optional[np.ndarray]:
    """能量（Riesz s=1）精修少量轮：能量最优构型通常也离覆盖最优很近。"""
    P, _ = so.thomson(Q, s=s, iters=30)
    return P


def kcenter_solve(P: np.ndarray, iters: int = 150, step: float = 0.25,
                  escape_rounds: int = 12) -> Tuple[np.ndarray, List[float]]:
    Q, hist = kcenter_local_search(P, iters=iters, step=step)
    R = so.coverage_radius(Q)
    for _ in range(escape_rounds):
        improved = False
        for maker in (
            lambda: _single_worst_step(Q, 0.02, 0.35),
            lambda: _lloyd_one(Q),
            lambda: _energy_polish_step(Q),
        ):
            try:
                cand = maker()
            except Exception:
                cand = None
            if cand is None:
                continue
            Rc = so.coverage_radius(cand)
            if Rc < R - 1e-14:
                # 候选更好：再从它出发做一轮 k-center
                Q2, _ = kcenter_local_search(cand, iters=iters, step=step)
                R2 = so.coverage_radius(Q2)
                if R2 < Rc:
                    Q2, Rc = Q2, R2
                else:
                    Q2 = cand
                Q, R = Q2, Rc
                improved = True
                break
        hist.append(R)
        if not improved:
            break
    return Q, hist
