"""覆盖目标（最小化精确 R_cov）的统一求解器。

思路：**把所有候选构型的精确 R_cov 算出来，取最小的那个**。
候选来自四个方向：
  1. 能量（Riesz s=1）多起点的解 —— 能量最优构型往往已经非常接近覆盖最优；
  2. Tammes（最大最小间距）的解；
  3. 旧球面 Lloyd 的解（最小化均方距离，作为多样性来源）；
  4. k-center 局部搜索（直接最小化 R_cov，带"并列最远点平均方向"与逃逸候选）。

按点数分档以控制耗时：
  * n ≤ 40 : 完整 k-center（多起点 + 逃逸候选），最容易拿到全局最优；
  * 40 < n ≤ 100 : 候选集 + 单起点 k-center + 逃逸；
  * n > 100 : 候选集 + 对最好的 3 个候选做短程 k-center 抛光（不做多起点）。

R_cov 是 O(n^2) 的精确计算，因此"取候选最小值"这件事本身是有保证的：
入选者的 R_cov 一定不差于任何被评估过的候选。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import sphoptim as so
from kcenter import kcenter_local_search
from kcenter_escape import kcenter_solve, _lloyd_one


def _R(P: np.ndarray) -> float:
    return so.coverage_radius(P)


def _dedup(cands: List[Tuple[str, np.ndarray]]) -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    for tag, P in cands:
        if P is None:
            continue
        dup = False
        for _, Q in out:
            if len(Q) == len(P) and float(np.abs(Q - P).max()) < 1e-9:
                dup = True
                break
        if not dup:
            out.append((tag, P))
    return out


def collect_candidates(n: int, seed: int = 0, rng: Optional[np.random.Generator] = None,
                       iters_energy: int = 300, iters_tammes: int = 6000,
                       iters_lloyd: int = 120) -> List[Tuple[str, np.ndarray]]:
    """生成一批候选构型（不评估 R_cov，交给调用者统一评估）。"""
    rng = rng or np.random.default_rng(seed)
    cands: List[Tuple[str, np.ndarray]] = []

    # 1) 能量多起点
    starts = [so.fibonacci_points(n),
              so.spiral_points(n, jitter=0.15, rng=rng),
              so.spiral_points(n, jitter=0.4, rng=rng)]
    for k, P0 in enumerate(starts):
        Pe, _ = so.thomson(P0, 1.0, iters=iters_energy)
        cands.append((f"energy{k}", Pe))
    # 2) Tammes
    try:
        Pt, _ = so.tammes(so.fibonacci_points(n), iters=iters_tammes, rate=0.05, seed=seed)
        cands.append(("tammes", Pt))
    except Exception:
        pass
    # 3) 球面 Lloyd
    for rate in (1.0, 0.8):
        Pl, _ = so.coverage_lloyd(so.fibonacci_points(n), iters=iters_lloyd,
                                  rate=rate, decay=0.99)
        cands.append((f"lloyd{rate}", Pl))
    return _dedup(cands)


def solve_cover(n: int, seed: int = 0, verbose: bool = False,
                energy_iters: int = 300, tammes_iters: int = 6000) -> Dict:
    """返回覆盖目标的最佳构型（按精确 R_cov 择优）。"""
    rng = np.random.default_rng(seed)
    cands = collect_candidates(n, seed=seed, rng=rng,
                              iters_energy=energy_iters, iters_tammes=tammes_iters)

    # 4) k-center 候选
    if n <= 40:
        for jit in (0.0, 0.25):
            P0 = so.fibonacci_points(n) if jit == 0 else so.spiral_points(n, jitter=jit, rng=rng)
            Q, _ = kcenter_solve(P0, iters=150, escape_rounds=12)
            cands.append((f"kcenter{jit}", Q))
    elif n <= 100:
        Q, _ = kcenter_solve(so.fibonacci_points(n), iters=60, escape_rounds=6)
        cands.append(("kcenter", Q))
        Q2, _ = kcenter_local_search(so.spiral_points(n, jitter=0.3, rng=rng), iters=60)
        cands.append(("kcenter_jit", Q2))
    else:
        # 大 N：先按精确 R_cov 挑出最好的候选，再做短程 k-center 抛光
        scored = sorted(((float(_R(P)), tag, P) for tag, P in cands), key=lambda x: x[0])
        for _, tag, P in scored[:3]:
            Q, _ = kcenter_local_search(P, iters=25, step=0.2)
            cands.append((f"kc_polish:{tag}", Q))

    scored = sorted(((float(_R(P)), tag, P) for tag, P in cands), key=lambda x: x[0])
    if verbose:
        for r, tag, _ in scored:
            print(f"      {tag:<18} R_cov={math.degrees(r):9.5f}°", flush=True)
    R, tag, P = scored[0]
    return {"points": so.normalize(P), "radius_deg": math.degrees(R), "tag": tag}
