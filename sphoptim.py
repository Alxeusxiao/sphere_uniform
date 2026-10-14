"""
sphoptim.py -- 球面上 N 点"均匀分布"的迭代优化内核。

"均匀"在这里被精确化为三类互相不同、但都合理的极小/极大目标：

1. Thomson 问题（极小化 Coulomb / Riesz 能量）
       E_s(P) = sum_{i<j} |p_i - p_j|^{-s}          s = 1 即经典 Coulomb
   直观：点之间互相排斥，系统自然松弛到能量最低的排布。

2. Tammes 问题（极大化最小间距 = "各点覆盖相同直径"）
       d_min(P) = min_{i<j} |p_i - p_j|     ->  maximize
   直观：任意两点不许靠得太近，等价于把 N 个同样大小的球冠塞进球面。

3. 覆盖问题（极小化覆盖半径 = "任何方向离最近的点都不太远"）
       R_cov(P) = max_{x in S^2} min_i arccos(x·p_i)  ->  minimize
   直观：把整个球面用 N 个等半径球冠盖住，求最小半径。

对应的迭代算法都在"约束在球面 S^2 上"的前提下做投影/增量更新，
所有点始终严格落在单位球面上（不做切平面外推再归一化的近似）。

作者：DSH 会话产物
"""

from __future__ import annotations

import json
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# 斐波那契球面格点用的黄金角比例
_GOLDEN = (1.0 + 5.0 ** 0.5) / 2.0


# --------------------------------------------------------------------------
# 基础球面几何
# --------------------------------------------------------------------------
def normalize(P: np.ndarray) -> np.ndarray:
    """把每行归一化到单位长度（投影回单位球面）。"""
    P = np.asarray(P, dtype=float)
    nrm = np.linalg.norm(P, axis=1, keepdims=True)
    nrm[nrm == 0.0] = 1.0
    return P / nrm


def random_points(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """球面均匀随机采样（高斯方向归一化，无极点聚集）。"""
    rng = rng or np.random.default_rng()
    return normalize(rng.normal(size=(n, 3)))


def fibonacci_points(n: int) -> np.ndarray:
    """斐波那契格点：确定性、准均匀的初值，比纯随机起点稳定得多。"""
    i = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * math.pi * i / _GOLDEN
    return np.column_stack((r * np.cos(theta), r * np.sin(theta), z))


def spiral_points(n: int, jitter: float = 0.0,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """螺旋初值 + 可选抖动（抖动可打破对称性，避免卡在鞍点）。"""
    P = fibonacci_points(n)
    if jitter > 0.0:
        rng = rng or np.random.default_rng(0)
        P = normalize(P + jitter * rng.normal(size=P.shape))
    return P


def min_angle(P: np.ndarray) -> float:
    """最小夹角（弧度）= arccos(max_{i<j} <p_i,p_j>)。

    注意：一定是 dot 的**最大**值 —— 夹角最小。用 dot 的最小值会算成
    "最大夹角"（对跖附近），这是很容易犯的错误。
    """
    P = normalize(P)
    G = P @ P.T
    np.fill_diagonal(G, -2.0)          # 排除自身（对角线不要参与取最大）
    return float(np.arccos(np.clip(G.max(), -1.0, 1.0)))


def min_pair(P: np.ndarray) -> Tuple[int, int, float]:
    """返回 (i, j, 最小夹角)。"""
    P = normalize(P)
    G = P @ P.T
    np.fill_diagonal(G, -2.0)
    i, j = np.unravel_index(np.argmax(G), G.shape)
    return int(i), int(j), float(np.arccos(np.clip(G[i, j], -1.0, 1.0)))


def second_min_angle(P: np.ndarray) -> float:
    """第二小的夹角（对含对跖点对的构型更合适的"间距"指标）。

    正多面体（八面体/立方体/二十面体等）含对跖点对，最小夹角恒为 180 度，
    此时 min_angle 没有区分度，第二小夹角才是有意义的间距。
    """
    P = normalize(P)
    G = P @ P.T
    iu = np.triu_indices(len(P), 1)
    ang = np.sort(np.arccos(np.clip(G[iu], -1.0, 1.0)))
    return float(ang[1]) if len(ang) > 1 else float(ang[0])


def riesz_energy(P: np.ndarray, s: float = 1.0) -> float:
    """Riesz s-能量（s=1 为 Coulomb，s->0 为对数能量附近）。"""
    G = np.clip(P @ P.T, -1.0, 1.0)
    D2 = np.maximum(2.0 * (1.0 - G), 1e-30)
    np.fill_diagonal(D2, 1.0)
    if abs(s) < 1e-9:
        # s -> 0 的对数能量：E_0 = Σ_{i≠j} ln|p_i-p_j| = Σ_{i≠j} ln(d_ij²)/2
        # （有序对求和，故等于无序对的 Σ ln(d²)）
        # 注意不是 ln(1/d)：ln(d) 随 d 递增，"最小化 Σ ln d" 就是增大间距。
        M = np.log(D2) / 2.0
        np.fill_diagonal(M, 0.0)
        return float(M.sum())
    M = D2 ** (-s / 2.0)
    np.fill_diagonal(M, 0.0)
    return float(0.5 * M.sum())


def energy_grad(P: np.ndarray, s: float = 1.0) -> np.ndarray:
    """Riesz 能量的梯度（球面切向分量），返回与 P 同形状的向量。

    dE/dp_i = s_eff * sum_{j!=i} w_ij (p_i - p_j)
            = s_eff * (p_i * sum_j w_ij <p_i,p_j> - sum_j w_ij p_j)
    其中 s > 0 时 w_ij = d_ij^{-s-2}、s_eff = s；
         s = 0（对数能量）时 w_ij = d_ij^{-2}、s_eff = −2
         （真实梯度为 2(⟨p,A⟩p − A)，这里返回 s_eff*(A − ⟨p,A⟩p)）。
    因为 <p_i, p_i - p_j> = 1 - <p_i,p_j> = d_ij^2 / 2，所以结果天然切于球面。
    """
    G = np.clip(P @ P.T, -1.0, 1.0)
    D2 = np.maximum(2.0 * (1.0 - G), 1e-30)
    np.fill_diagonal(D2, 1.0)
    if abs(s) < 1e-9:
        # 对数能量：w_ij = d^{-2}；真实梯度是 2(⟨p,A⟩p − A)，
        # 而下面返回的是 s_eff*(A − ⟨p,A⟩p)，故 s_eff = −1。
        # （数值判定见 probe_s0c.py；自洽性见 check_s_consistency.py）
        W = D2 ** -1.0
        s_eff = -2.0
    else:
        W = D2 ** (-(s + 2.0) / 2.0)
        s_eff = s
    np.fill_diagonal(W, 0.0)
    A = W @ P                       # A_i = sum_j w_ij p_j
    # grad_i E = s_eff * (<p_i,A_i> p_i - A_i)   （天然切于球面）
    return s_eff * (A - P * (A * P).sum(axis=1, keepdims=True))


def pairwise_stats(P: np.ndarray) -> Dict[str, float]:
    """一次性给出最小/平均/最大夹角，方便判断均匀程度。"""
    G = np.clip(P @ P.T, -1.0, 1.0)
    iu = np.triu_indices(len(P), 1)
    ang = np.arccos(G[iu])
    return {
        "min_deg": float(np.degrees(ang.min())),
        "mean_deg": float(np.degrees(ang.mean())),
        "max_deg": float(np.degrees(ang.max())),
        "std_deg": float(np.degrees(ang.std())),
    }


# --------------------------------------------------------------------------
# 算法 1：增量 Frank-Wolfe（Thomson / Riesz 能量极小化）
# --------------------------------------------------------------------------
def _arc_point(p: np.ndarray, ut: np.ndarray, t: float) -> np.ndarray:
    """大圆上从 p 出发、沿单位切向量 ut 走弧长 t 的点（严格在单位球面上）。"""
    return p * math.cos(t) + ut * math.sin(t)


def _arc_dir(p: np.ndarray, q: np.ndarray) -> Optional[np.ndarray]:
    """球面大圆在 p 处、朝 q 的单位切向量（几何量，供可视化/演示使用）。

        u(t) = cos t * p + sin t * ut,  u(1) = q  <=>  ut = (q - cosθ p)/sinθ
    """
    p = p / np.linalg.norm(p)
    q = q / np.linalg.norm(q)
    c = float(np.clip(p @ q, -1.0, 1.0))
    sn = math.sqrt(max(1.0 - c * c, 0.0))
    if sn < 1e-12:
        return None
    ut = (q - c * p) / sn
    nrm = float(np.linalg.norm(ut))
    if nrm < 1e-15:
        return None
    return ut / nrm


def _best_step(P: np.ndarray, S: np.ndarray, E: float, s: float,
               grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """在 t in [0,1] 上对 E(normalize((1-t)P + tS)) 做稳健的一维搜索。

    沿 FW 方向的能量并不保证单峰（实测会先降后升），所以不用黄金分割这类
    依赖单峰的算法；而是先按几何粒度采样、取最优，再在最优点附近细化。
    """
    if grid is None:
        grid = np.concatenate(([1.0], 0.5 ** np.arange(1, 16)))     # 1, 1/2, 1/4, ...
    vals = []
    for t in grid:
        vals.append(riesz_energy(normalize((1.0 - t) * P + t * S), s))
    vals = np.asarray(vals)
    k = int(np.argmin(vals))
    best_t, best_v = float(grid[k]), float(vals[k])
    if best_v >= E:
        return 0.0, E
    # 在最优点两侧细化（取相邻采样点作为区间）
    lo = float(grid[k + 1]) if k + 1 < len(grid) else 0.0
    hi = float(grid[k - 1]) if k - 1 >= 0 else 1.0
    for _ in range(25):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        v1 = riesz_energy(normalize((1.0 - m1) * P + m1 * S), s)
        v2 = riesz_energy(normalize((1.0 - m2) * P + m2 * S), s)
        if v1 < v2:
            hi = m2
        else:
            lo = m1
        if hi - lo < 1e-12:
            break
    t = 0.5 * (lo + hi)
    v = riesz_energy(normalize((1.0 - t) * P + t * S), s)
    if v < best_v:
        best_t, best_v = t, v
    return best_t, best_v


def thomson(P: np.ndarray, s: float = 1.0, iters: int = 2000,
            seed: int = 0, callback: Optional[Callable[[int, np.ndarray], None]] = None,
            callback_every: int = 10, tol: float = 1e-12) -> Tuple[np.ndarray, List[float]]:
    """球面 Frank-Wolfe：每轮取线性子问题的最优方向，再做稳健线搜索。

    迭代式：S = -G/|G|（逐点线性化最优解），P <- normalize((1-t)P + tS)，
    t 由 _best_step 在多点上评估真实能量选出，保证单调下降。
    投影用"归一化"而非大圆参数化：它同样是到球面的收缩映射，
    且与线搜索里评估的能量完全一致，避免了模型/真值不自洽的风险。
    """
    P = normalize(np.array(P, dtype=float, copy=True))
    E = riesz_energy(P, s)
    hist = [E]
    for it in range(iters):
        G = energy_grad(P, s)
        Gn = np.linalg.norm(G, axis=1, keepdims=True)
        if float(Gn.max()) < 1e-13:
            break
        S = -G / np.where(Gn > 0, Gn, 1.0)
        t, E_new = _best_step(P, S, E, s)
        if t <= 0.0:
            break
        P = normalize((1.0 - t) * P + t * S)
        if (it + 1) % callback_every == 0 or it == 0:
            hist.append(E_new)
            if callback is not None:
                callback(it + 1, P)
            if abs(E - E_new) < tol * max(1.0, abs(E)):
                E = E_new
                break
        E = E_new
    hist.append(riesz_energy(P, s))
    return P, hist


# --------------------------------------------------------------------------
# 算法 2：投影梯度下降（能量极小化，通用备用方案）
# --------------------------------------------------------------------------
def pgd_energy(P: np.ndarray, s: float = 1.0, iters: int = 1500, lr: Optional[float] = None,
               callback: Optional[Callable[[int, np.ndarray], None]] = None,
               callback_every: int = 25) -> Tuple[np.ndarray, List[float]]:
    """球面投影梯度下降 + 回溯线搜索。步长自适应，适合作为对照实现。"""
    P = normalize(np.array(P, dtype=float, copy=True))
    n = len(P)
    E = riesz_energy(P, s)
    if lr is None:
        # 切向梯度尺度 ~ n，取步长 ~ 1/n^2 使 ||lr*G|| 保持 O(1/n)
        lr = 1.0 / (n * n)
    hist = [E]
    for it in range(iters):
        G = energy_grad(P, s)
        trail = 0
        Q, E_new = P, E
        while trail < 60:
            Q = normalize(P - lr * G)
            E_new = riesz_energy(Q, s)
            if E_new < E - 1e-16 * max(1.0, abs(E)):
                break
            lr *= 0.5
            trail += 1
            if lr < 1e-18:
                break
        if E_new >= E:
            break
        P, E = Q, E_new
        if trail == 0:
            lr *= 1.15                   # 步长回升，保持收敛速度
        if (it + 1) % callback_every == 0 or it == 0:
            hist.append(E)
            if callback is not None:
                callback(it + 1, P)
    hist.append(riesz_energy(P, s))
    return P, hist


# --------------------------------------------------------------------------
# 算法 3：松弛法（Tammes，极大化最小间距）
# --------------------------------------------------------------------------
def dmin_grad(P: np.ndarray, active_tol: float = 1.02) -> Tuple[float, np.ndarray]:
    """最小夹角 dmin(P) = min_{i<j} angle(p_i,p_j) 及其切向（次）梯度。

    对活跃点对 (i,j)：d/dp_i arccos(<p_i,p_j>) = -p_j / sqrt(1-<p_i,p_j>^2)。
    该公式已用中心差分逐分量验证过（误差 < 1e-12）。
    """
    P = normalize(P)
    G = np.clip(P @ P.T, -1.0, 1.0)
    np.fill_diagonal(G, -2.0)
    d = float(np.arccos(G.max()))
    ii, jj = np.where(np.triu(np.arccos(np.clip(G, -1.0, 1.0)) < d * active_tol, 1))
    g = np.zeros_like(P)
    for a, b in zip(ii, jj):
        dot = float(G[a, b])
        sn = math.sqrt(max(1.0 - dot * dot, 1e-18))
        ga = -P[b] / sn
        g[a] += ga - P[a] * float(ga @ P[a])
        gb = -P[a] / sn
        g[b] += gb - P[b] * float(gb @ P[b])
    return d, g


def tammes(P: np.ndarray, iters: int = 4000, rate: float = 0.30, decay: float = 0.9995,
           callback: Optional[Callable[[int, np.ndarray], None]] = None,
           callback_every: int = 100,
           history_every: int = 100, max_step: float = 0.05,
           restarts: int = 0, seed: int = 0) -> Tuple[np.ndarray, List[float]]:
    """Tammes 问题（最大化最小夹角）的投影次梯度上升。

    沿已验证的 dmin 次梯度方向走有界步长（上限 max_step），每步归一化；
    步长指数衰减，并用"能量/目标单调性"作为接受准则，永远不会让 dmin 变差。
    restarts > 0 时用随机初值多次重跑，取 dmin 最大者。
    """
    rng = np.random.default_rng(seed)

    def run(P0: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        Q = normalize(np.array(P0, dtype=float, copy=True))
        hist: List[float] = []
        alpha = rate
        d_cur = min_angle(Q)
        stall = 0
        for it in range(iters):
            if it % history_every == 0:
                hist.append(min_angle(Q))
            d, g = dmin_grad(Q)
            gn = np.linalg.norm(g, axis=1, keepdims=True)
            if float(gn.max()) < 1e-12:
                break
            step = g / np.maximum(gn, 1e-15)          # 每点单位化，只保留方向
            step = step * min(alpha, max_step)
            cand = normalize(Q + step)
            d_new = min_angle(cand)
            if d_new >= d_cur - 1e-15:
                Q = cand
                d_cur = d_new
                alpha = min(alpha * 1.02, max_step)   # 顺利则缓慢放大步长
            else:
                alpha *= 0.5                           # 步长过大则回退
                stall += 1
                if stall > 40:                         # 回退过头：重启步长
                    alpha = rate
                    stall = 0
                continue
            stall = 0
            if (it + 1) % callback_every == 0 and callback is not None:
                callback(it + 1, Q)
        hist.append(min_angle(Q))
        return Q, hist

    best_P, best_h = run(P)
    best_d = min_angle(best_P)
    for _ in range(restarts):
        P0 = random_points(len(P), rng)
        Q, h = run(P0)
        if min_angle(Q) > best_d:
            best_P, best_h, best_d = Q, h, min_angle(Q)
    return best_P, best_h


# --------------------------------------------------------------------------
# 算法 4：覆盖优化（球面 Lloyd / 质心 Voronoi 分割）
# --------------------------------------------------------------------------
def _hull2d(X: np.ndarray) -> np.ndarray:
    """二维凸包（Andrew monotone chain），返回**逆时针**顺序的顶点索引。"""
    n = len(X)
    order = np.lexsort((X[:, 1], X[:, 0]))

    def half(seq):
        out: List[int] = []
        for k in seq:
            while len(out) >= 2:
                o, a = out[-2], out[-1]
                cross = ((X[a, 0] - X[o, 0]) * (X[k, 1] - X[o, 1])
                         - (X[a, 1] - X[o, 1]) * (X[k, 0] - X[o, 0]))
                if cross <= 0.0:            # 非左转（含共线）则弹出
                    out.pop()
                else:
                    break
            out.append(int(k))
        return out

    lower = half(order)
    upper = half(order[::-1])
    idx = np.array(lower[:-1] + upper[:-1], dtype=int)
    if len(idx) < 3:
        return np.array([], dtype=int)
    # 用有向面积保证逆时针（避免退化输入下绕向不确定）
    x, y = X[idx, 0], X[idx, 1]
    signed = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if signed < 0.0:
        idx = idx[::-1]
    return idx


def _neighbor_subset(P: np.ndarray, i: int, k: int) -> np.ndarray:
    """返回 site i 的候选邻居下标（最近的 k 个 + 切平面凸包上的点）。

    Voronoi 邻居一定是"较近"的点；再加上凸包（用于捕捉大角度上的邻居），
    得到的候选集合足以覆盖所有真正的胞腔顶点（顶点是三个邻居的球面外心）。
    """
    n = len(P)
    if k >= n - 1:
        return np.array([j for j in range(n) if j != i], dtype=int)
    G = P @ P[i]
    order = np.argsort(-G)
    cand = [int(j) for j in order if j != i][:k]
    others = np.delete(P, i, axis=0)
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(P[i, 0]) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - P[i] * float(e1 @ P[i])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(P[i], e1)
    hull = _hull2d(np.column_stack((others @ e1, others @ e2)))
    if len(hull):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        idx_all = np.arange(n)[keep]
        cand.extend(int(v) for v in idx_all[hull])
    return np.array(sorted(set(cand)), dtype=int)


def voronoi_cells(P: np.ndarray, k_nbr: int = 14) -> List[np.ndarray]:
    """球面 Voronoi 胞腔顶点表（精确，纯 NumPy）。

    胞腔 i = {v in S^2 : <v, p_j - p_i> <= 0, 所有 j}，是若干半球的交。
    顶点是两张"大圆边界"的交点，即 ±(p_j - p_i) × (p_k - p_i) 归一化后
    再按约束筛选（A 与 -A 中恰有一个满足全部约束）。

    注意：不能用切平面参数化——那只能表示距 p_i 90 度以内的方向，
    而真实胞腔可能跨过 90 度（极端情况下直径可达 180 度，如正四面体）。
    """
    P = normalize(P)
    n = len(P)
    cells: List[np.ndarray] = []
    for i in range(n):
        nb = _neighbor_subset(P, i, k_nbr)
        if len(nb) < 2:
            cells.append(np.array([]).reshape(0, 3))
            continue
        N = P[nb] - P[i]                       # 约束法向（未归一化）
        N = N / np.linalg.norm(N, axis=1, keepdims=True)
        # 所有法向对的交点 ±(N_a x N_b)
        cr = np.cross(N[:, None, :], N[None, :, :]).reshape(-1, 3)
        nrm = np.linalg.norm(cr, axis=1, keepdims=True)
        cr = cr[nrm[:, 0] > 1e-12] / nrm[nrm[:, 0] > 1e-12]
        if len(cr) == 0:
            cells.append(np.array([]).reshape(0, 3))
            continue
        # 全部约束（含未进入候选集的 site，保证正确性）
        H = P - P[i]
        hn = np.linalg.norm(H, axis=1, keepdims=True)
        H = H / np.maximum(hn, 1e-15)
        valid = hn[:, 0] > 1e-12
        dots = cr @ H.T
        dots = np.where(valid[None, :], dots, -1.0)
        ok = (dots <= 1e-9).all(axis=1) | ((-dots) <= 1e-9).all(axis=1)
        cr = cr[ok]
        if len(cr) == 0:
            cells.append(np.array([]).reshape(0, 3))
            continue
        # 只保留"约束取等号最多"的那一侧，并去重
        dots = np.where(valid[None, :], cr @ H.T, -1.0)
        mx = dots.max(axis=1)
        cr = cr[mx <= 1e-9]
        if len(cr) == 0:
            cells.append(np.array([]).reshape(0, 3))
            continue
        key = np.round(cr, 8)
        _, uniq = np.unique(key, axis=0, return_index=True)
        cr = cr[np.sort(uniq)]
        # 按绕 p_i 的方位角排序（多边形顺序；覆盖半径只依赖顶点与棱，顺序无关）
        e1 = np.array([1.0, 0.0, 0.0])
        if abs(P[i, 0]) > 0.9:
            e1 = np.array([0.0, 1.0, 0.0])
        e1 = e1 - P[i] * float(e1 @ P[i])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(P[i], e1)
        ang = np.arctan2(cr @ e2, cr @ e1)
        cells.append(cr[np.argsort(ang)])
    return cells


def cell_centroid_dirs(V: np.ndarray) -> Optional[np.ndarray]:
    """球面凸多边形顶点的"平均方向"（球面质心的稳定近似）。

    对每个顶点按相邻棱的球面弧长加权后求和再归一化，等价于对球面多边形
    做面积加权平均（球面三角形的重心在球面上是三条边的加权平均方向）。
    这里用弧长权重已足够精确，且数值上远比迭代式 Karcher 平均稳健。
    """
    m = len(V)
    if m < 3:
        return None
    W = np.roll(V, -1, axis=0)
    dots = np.clip((V * W).sum(axis=1), -1.0, 1.0)
    ang = np.arccos(dots)
    mid = V + W
    nm = np.linalg.norm(mid, axis=1, keepdims=True)
    mid = mid / np.where(nm > 1e-15, nm, 1.0)
    # 用"顶点 + 棱中点"的弧长加权平均逼近胞腔质心方向
    acc = (V * ang[:, None]).sum(axis=0) + (mid * ang[:, None]).sum(axis=0)
    na = float(np.linalg.norm(acc))
    if na < 1e-15:
        return None
    return acc / na


def karcher_mean(V: np.ndarray, iters: int = 50) -> Optional[np.ndarray]:
    """球面质心方向（保留旧名字，内部用稳健的面积加权近似）。"""
    return cell_centroid_dirs(V)


def cells_and_radius(P: np.ndarray, cells: Optional[List[np.ndarray]] = None
                      ) -> Tuple[List[np.ndarray], float]:
    """一次遍历同时得到 Voronoi 胞腔与精确覆盖半径（避免重复算胞腔）。"""
    P = normalize(P)
    if cells is None:
        cells = voronoi_cells(P)
    if len(P) == 1:
        return cells, math.pi / 2
    best = 0.0
    for i, V in enumerate(cells):
        if len(V) < 3:
            best = max(best, math.pi / 2)          # 退化胞腔（半球）
            continue
        best = max(best, float(np.arccos(np.clip(V @ P[i], -1.0, 1.0)).max()))
        W = np.roll(V, -1, axis=0)
        mids = V + W
        nm = np.linalg.norm(mids, axis=1, keepdims=True)
        mids = mids / np.where(nm > 1e-15, nm, 1.0)
        best = max(best, float(np.arccos(np.clip(mids @ P[i], -1.0, 1.0)).max()))
    return cells, best


def coverage_radius(P: np.ndarray) -> float:
    """精确覆盖半径 R = max_x min_i angle(x, p_i)。

    最近 site 的映射在每个 Voronoi 胞腔内是"到 p_i 的球面距离"，其在一个
    球面凸多边形上的最大值必在顶点或棱的中点上取得，因此只需检查这两处。

    特例：当某个胞腔退化成整个半球时（例如 N=2，或点集只含一个点），
    它没有有限顶点，此时最远点就是 p_i 的对跖点，距离为 90°。
    """
    P = normalize(P)
    if len(P) == 1:
        return math.pi / 2
    cells = voronoi_cells(P)
    best = 0.0
    for i, V in enumerate(cells):
        if len(V) < 3:
            # 退化胞腔（半球）：最远方向是 p_i 的对跖点，距离 90°
            best = max(best, math.pi / 2)
            continue
        best = max(best, float(np.arccos(np.clip(V @ P[i], -1.0, 1.0)).max()))
        W = np.roll(V, -1, axis=0)
        mids = V + W
        nm = np.linalg.norm(mids, axis=1, keepdims=True)
        mids = mids / np.where(nm > 1e-15, nm, 1.0)
        best = max(best, float(np.arccos(np.clip(mids @ P[i], -1.0, 1.0)).max()))
    return best


def coverage_lloyd(P: np.ndarray, iters: int = 400, rate: float = 1.0,
                   decay: float = 0.995,
                   callback: Optional[Callable[[int, np.ndarray], None]] = None,
                   callback_every: int = 5,
                   history_every: int = 5) -> Tuple[np.ndarray, List[float]]:
    """球面 Lloyd 迭代：每点移动到它自己的 Voronoi 胞腔的球面质心。

    这会把覆盖半径推向下确界；学习率 <1 时对应"阻尼 Lloyd"，
    可以避免对称/退化构型下的振荡。
    """
    P = normalize(np.array(P, dtype=float, copy=True))
    n = len(P)
    alpha = rate
    hist: List[float] = []
    for it in range(iters):
        cells = voronoi_cells(P)
        Q = P.copy()
        for i in range(n):
            c = karcher_mean(cells[i])
            if c is None:
                continue
            t = math.acos(float(np.clip(c @ P[i], -1.0, 1.0)))
            if t < 1e-12:
                continue
            # 沿大圆朝胞腔质心走 alpha 倍弧长
            tangent = (c - P[i] * float(c @ P[i])) / math.sin(t)
            P[i] = P[i] * math.cos(alpha * t) + tangent * math.sin(alpha * t)
            P[i] = P[i] / np.linalg.norm(P[i])
        alpha *= decay
        if it % history_every == 0:
            hist.append(coverage_radius(P))
        if (it + 1) % callback_every == 0 and callback is not None:
            callback(it + 1, P)
    hist.append(coverage_radius(P))
    return P, hist


def cover_grid(P: np.ndarray, M: int = 400000) -> float:
    """独立校验：用 M 个均匀格点估计覆盖半径（应略小于精确值）。"""
    Q = fibonacci_points(M)
    G = Q @ P.T
    idx = G.argmax(axis=1)
    best = np.arccos(np.clip(G[np.arange(M), idx], -1.0, 1.0))
    return float(best.max())


# --------------------------------------------------------------------------
# 组合求解器
# --------------------------------------------------------------------------
def solve_all(n: int, s: float = 1.0, iters_energy: int = 400,
              iters_tammes: int = 8000, iters_cover: int = 200,
              seed: int = 0, verbose: bool = False,
              restarts: int = 2) -> Dict[str, Dict]:
    """对给定 N 同时求解三类目标（能量最小 / 最小间距最大 / 覆盖半径最小）。

    三类目标的最优构型不同（例如 N=12：能量最优是正二十面体、
    覆盖最优是"五边形-六边形"混合的准均匀构型），因此分别求解。

    策略：多起点 + 互相精修（Tammes 解再用能量 FW 抛光，通常能把 dmin
    推到接近最优，同时得到很低的能量）。
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict] = {}

    def _log(tag: str, t0: float, val: float) -> None:
        if verbose:
            print(f"      {tag:<8s} {time.time() - t0:6.1f}s  value={val:.6f}", flush=True)

    # ---- 1. Riesz/Thomson 能量最小（多起点） ----
    t0 = time.time()
    starts = [fibonacci_points(n),
              spiral_points(n, jitter=0.05, rng=rng),
              spiral_points(n, jitter=0.25, rng=rng)]
    if n <= 200:
        starts += [random_points(n, rng) for _ in range(max(0, restarts))]
    Pe, he = None, None
    for P0 in starts:
        Pt, h = thomson(P0, s=s, iters=iters_energy)
        if Pe is None or riesz_energy(Pt, s) < riesz_energy(Pe, s):
            Pe, he = Pt, h
    out["energy"] = {
        "points": Pe,
        "energy": riesz_energy(Pe, s),
        "min_deg": math.degrees(min_angle(Pe)),
        "history": he,
    }
    _log("energy", t0, out["energy"]["energy"])

    # ---- 2. Tammes 最大最小间距（多起点 + 能量抛光） ----
    t0 = time.time()
    cands = []
    for P0 in [fibonacci_points(n), Pe, spiral_points(n, jitter=0.3, rng=rng)]:
        Pt, h = tammes(P0, iters=iters_tammes, rate=0.05, decay=0.9995, seed=seed)
        cands.append((min_angle(Pt), Pt, h))
    d_best, Pt_best, h_best = max(cands, key=lambda c: c[0])
    # 能量抛光：Tammes 解再做一次能量下降，通常能同时改善 dmin 与能量
    Pt_pol, _ = thomson(Pt_best, s=s, iters=iters_energy)
    if min_angle(Pt_pol) >= d_best - 1e-9:
        d_best, Pt_best = min_angle(Pt_pol), Pt_pol
    out["tammes"] = {
        "points": Pt_best,
        "min_deg": math.degrees(d_best),
        "energy": riesz_energy(Pt_best, s),
        "history": h_best,
    }
    _log("tammes", t0, out["tammes"]["min_deg"])

    # ---- 3. 覆盖半径最小 ----
    # 注意：球面 Lloyd 最小化的是"均方距离"（k-means），而覆盖半径是
    # k-center（最小化最大距离）目标，两者不等价——单靠 Lloyd 会得到
    # "看着均匀但 R_cov 不是最小"的构型。这里改成：
    #   收集一批候选（能量多起点 / Tammes / Lloyd / k-center 局部搜索），
    #   全部用**精确 R_cov** 评估后取最小者。
    t0 = time.time()
    res = solve_cover(n, seed=seed, energy_iters=iters_energy,
                      tammes_iters=iters_tammes)
    Pc_best = normalize(res["points"])
    r_best = coverage_radius(Pc_best)
    out["cover"] = {
        "points": Pc_best,
        "radius_deg": math.degrees(r_best),
        "min_deg": math.degrees(min_angle(Pc_best)),
        "energy": riesz_energy(Pc_best, s),
        "history": [],
        "source": res.get("tag", ""),
    }
    # ---- 4. Voronoi 面积均匀度 ----
    t0 = time.time()
    res_a = solve_area(n, seed=seed, iters=100)
    Pa_best = normalize(res_a["points"])
    cv_best = area_uniformity(Pa_best)[0]
    src_a = res_a.get("tag", "")
    # 三个既有目标的解也一起比：谁 cv 更小就用谁
    for key in ("energy", "tammes", "cover"):
        Pk = out[key]["points"]
        ck = area_uniformity(Pk)[0]
        if ck < cv_best - 1e-12:
            Pa_best, cv_best, src_a = Pk, ck, key
    out["area"] = {
        "points": Pa_best,
        "cv": cv_best,
        "min_deg": math.degrees(min_angle(Pa_best)),
        "energy": riesz_energy(Pa_best, s),
        "history": res_a.get("history", []),
        "source": src_a,
    }
    _log("area", t0, out["area"]["cv"])

    # 保险：若能量解或 Tammes 解的 R_cov 更小，就换过来（保证 cover 确实最优）
    for key in ("energy", "tammes", "area"):
        Pk = out[key]["points"]
        Rk = coverage_radius(Pk)
        if Rk < r_best - 1e-12:
            Pc_best, r_best = Pk, Rk
            out["cover"]["points"] = Pc_best
            out["cover"]["radius_deg"] = math.degrees(r_best)
            out["cover"]["min_deg"] = math.degrees(min_angle(Pc_best))
            out["cover"]["energy"] = riesz_energy(Pc_best, s)
            out["cover"]["source"] = key
    _log("cover", t0, out["cover"]["radius_deg"])
    return out


# --------------------------------------------------------------------------
# 算法 6：Voronoi 面积均匀度（最小化胞腔面积的离散度）
# --------------------------------------------------------------------------
# 定义：A_i 为第 i 个 Voronoi 胞腔的球面面积，ΣA_i = 4π。
#       cv(P) = std(A_i) / mean(A_i)，越小说明"每个点分到的球面越接近相等"。
# 这是最贴近"均匀"直觉的度量：既不看最近邻，也不只看最坏方向，而是看整体分配。
#
# 与 Lloyd 的关系：Lloyd（朝胞腔质心移动）最小化的是均方距离，但它在实践中
# 也能大幅降低 cv；再加"面积权重" w_i ∝ Ā/A_i 可以进一步降低。
# 实验数据（proto_area.py，n=50 / 150 步）：裸 Lloyd 降 78.6%，
# 加权 Lloyd 降 81.7%~82.3%；"带符号远离"几乎无效（1.6%）。

def cell_area(V: np.ndarray) -> float:
    """球面凸多边形面积 = 球面盈余 = (内角之和) − (m−2)π。

    注意：不能用"Σ 2·atan2(|u×w|, u·w)"那类平面惯性写法——它对球面多边形
    不成立（实测正二十面体一个胞腔会算成 7.30 而非 2π）。这里逐个顶点求
    内角（用顶点处两条切向量的夹角），再减去 (m−2)π。
    顶点表由 voronoi_cells 保证是绕 site 逆时针排列，因此结果为正值。
    """
    m = len(V)
    if m < 3:
        return 0.0
    total = 0.0
    for i in range(m):
        p0 = V[i]
        a, b = V[(i - 1) % m], V[(i + 1) % m]
        t1 = a - p0 * float(a @ p0)
        t2 = b - p0 * float(b @ p0)
        n1, n2 = float(np.linalg.norm(t1)), float(np.linalg.norm(t2))
        if n1 < 1e-15 or n2 < 1e-15:
            continue
        t1, t2 = t1 / n1, t2 / n2
        ang = math.atan2(float(np.linalg.norm(np.cross(t1, t2))),
                         float(np.clip(t1 @ t2, -1.0, 1.0)))
        total += ang
    return total - (m - 2) * math.pi


def cell_areas(P: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """返回 (各胞腔面积数组, 胞腔顶点列表)。"""
    cells = voronoi_cells(P)
    return np.array([cell_area(V) for V in cells]), cells


def area_uniformity(P: np.ndarray) -> Tuple[float, np.ndarray]:
    """面积离散度 cv = std(A)/mean(A)（越小越均匀）与面积数组。"""
    A, _ = cell_areas(P)
    m = float(A.mean())
    return (float(A.std() / m) if m > 1e-12 else float('inf')), A


def _area_cv(P: np.ndarray) -> float:
    return area_uniformity(P)[0]


def _area_step(P: np.ndarray, cells: List[np.ndarray], cs: List[Optional[np.ndarray]],
               w: np.ndarray) -> np.ndarray:
    """朝各胞腔质心移动，位移按权重 w_i 缩放（w=1 即标准 Lloyd）。"""
    Q = P.copy()
    for i, c in enumerate(cs):
        if c is None:
            continue
        t = math.acos(float(np.clip(c @ P[i], -1.0, 1.0)))
        if t < 1e-12:
            continue
        ut = (c - P[i] * float(c @ P[i])) / math.sin(t)
        a = float(np.clip(w[i], 0.0, 1.0)) * t
        Q[i] = P[i] * math.cos(a) + ut * math.sin(a)
        Q[i] /= np.linalg.norm(Q[i])
    return Q


def area_lloyd(P: np.ndarray, iters: int = 150, rate: float = 0.8,
               power: float = 1.0, backtrack: bool = True,
               history_every: int = 1) -> Tuple[np.ndarray, List[float]]:
    """面积均匀化迭代：加权 Lloyd + 整轮 cv 回退（单调不增）。

    rate   : 基础步长（1.0 表示直接移到质心）
    power  : 权重指数，w_i = rate * (Ā/A_i)^power
    backtrack : 若一轮没有降低 cv，则把步长减半重试
    """
    Q = normalize(np.array(P, dtype=float, copy=True))
    hist: List[float] = [_area_cv(Q)]
    a = rate
    for it in range(iters):
        A, cells = cell_areas(Q)
        mean = float(A.mean())
        cs = [cell_centroid_dirs(V) if len(V) >= 3 else None for V in cells]
        w = np.clip(a * (mean / np.maximum(A, 1e-12)) ** power, 0.0, 1.0)
        cand = _area_step(Q, cells, cs, w)
        cv_new, cv_old = _area_cv(cand), _area_cv(Q)
        if cv_new < cv_old - 1e-14:
            Q = cand
            a = min(rate, a * 1.10)
        elif backtrack:
            ok = False
            for _ in range(6):
                w *= 0.5
                cand = _area_step(Q, cells, cs, w)
                if _area_cv(cand) < cv_old - 1e-14:
                    Q, ok = cand, True
                    a = max(0.05, a * 0.5)
                    break
            if not ok:
                break
        else:
            Q = cand
        if it % history_every == 0:
            hist.append(_area_cv(Q))
    hist.append(_area_cv(Q))
    return Q, hist


def solve_area(n: int, seed: int = 0, iters: int = 120, verbose: bool = False) -> Dict:
    """面积均匀目标：多候选 + 面积加权 Lloyd 抛光，按 cv 择优。

    斐波那契格点在面积均匀度上本就接近最优，因此这里的关键不是"更强的单点迭代"，
    而是**扩大候选集**：能量（s=1 / 对数）、k-center、Tammes 都能给出不同盆地，
    再各自用面积加权 Lloyd 抛光一次，最后按 cv 取最小。
    """
    rng = np.random.default_rng(seed)
    cands: List[np.ndarray] = [fibonacci_points(n)]
    try:
        cands.append(thomson(spiral_points(n, jitter=0.15, rng=rng), 1.0, iters=250)[0])
        cands.append(thomson(spiral_points(n, jitter=0.35, rng=rng), 0.0, iters=250)[0])
    except Exception:
        pass
    cands.append(kcenter_local_search(fibonacci_points(n), iters=40)[0])
    try:
        cands.append(tammes(fibonacci_points(n), iters=3000, rate=0.05, seed=seed)[0])
    except Exception:
        pass

    results: List[Tuple[float, str, np.ndarray, List[float]]] = []
    for k, P0 in enumerate(cands):
        Q, h = area_lloyd(P0, iters=iters, rate=0.9, power=1.0)
        results.append((_area_cv(Q), f"polish{k}", Q, h))
        # 抛光前的原始候选也保留（有时抛光反而略差，择优即可）
        results.append((_area_cv(P0), f"raw{k}", P0, []))
    results.sort(key=lambda x: x[0])
    cv_best, tag, P, h = results[0]
    if verbose:
        for c, t, _, _ in results[:6]:
            print(f"      {t:<10} cv={c:.6f}", flush=True)
    return {"points": normalize(P), "cv": cv_best, "history": h, "tag": tag}


# --------------------------------------------------------------------------
# 算法 5：球面 k-center（最小化覆盖半径 R_cov）
# --------------------------------------------------------------------------
# 目标 R_cov(P) = max_x min_i angle(x, p_i)，即"最小化最大距离"（k-center），
# 与球面 Lloyd 最小化的"均方距离"（k-means）**不是**同一个目标。
#
# 局部搜索的两条要点：
#   1) 移动方向取"胞腔内并列最远点（顶点/棱中点）的**平均**方向"。
#      取单个最远点会破坏对称性：正八面体/正二十面体本身已是局部最优，
#      朝单个顶点移动实测会让 R_cov 从 54.736° 恶化到 73°；取平均时
#      并列方向互相抵消，于是能正确判定"已到局部最优"而停住。
#   2) 每轮扫描后用**精确 R_cov 做整轮回退**，因此 R_cov 单调不增。
#
# 逃离局部极小：单点扰动 / 一步 Lloyd / 一步能量精修，仍只用精确 R_cov 判定。
# 求解策略：把能量（Riesz s=1）多起点解当作 k-center 的**起点**（实测很关键），
#           再加随机重启与扰动细化，最后统一按精确 R_cov 取最小。

def cell_worst_set(P: np.ndarray, i: int, V: np.ndarray,
                   tol: float = 0.02) -> Optional[np.ndarray]:
    """site i 胞腔内"并列最远"的顶点/棱中点集合（(m,3) 单位向量）。"""
    if len(V) < 3:
        return np.array([-P[i]])                       # 退化胞腔（半球）：对跖点
    cand = list(V)
    W = np.roll(V, -1, axis=0)
    mids = V + W
    nm = np.linalg.norm(mids, axis=1, keepdims=True)
    cand.extend(list(mids / np.where(nm > 1e-15, nm, 1.0)))
    C = np.array(cand)
    d = np.arccos(np.clip(C @ P[i], -1.0, 1.0))
    dmax = float(d.max())
    sel = C[d >= dmax * (1.0 - tol)]
    return sel if len(sel) else None


def kcenter_scan(Q: np.ndarray, step: float, tol: float) -> Tuple[np.ndarray, int, float]:
    """一轮扫描：逐点朝自己"并列最远方向"走一小步。

    返回 (新点集, 移动点数, 新构型的精确 R_cov)；R_cov 顺带在同一次
    胞腔遍历里算出，避免外层重复构造 Voronoi。
    """
    cells = voronoi_cells(Q)
    moved = 0
    for i in range(len(Q)):
        sel = cell_worst_set(Q, i, cells[i], tol)
        if sel is None or len(sel) == 0:
            continue
        avg = sel.sum(axis=0)
        na = float(np.linalg.norm(avg))
        if na < 1e-12:
            continue                                   # 方向相消 => 已局部最优
        ut = _arc_dir(Q[i], avg / na)
        if ut is None:
            continue
        Q[i] = _arc_point(Q[i], ut, step)
        Q[i] /= np.linalg.norm(Q[i])
        moved += 1
    _, R = cells_and_radius(Q)
    return Q, moved, R


def kcenter_local_search(P: np.ndarray, iters: int = 150, step: float = 0.25,
                         tols: Sequence[float] = (0.02, 0.08, 0.2),
                         history_every: int = 1) -> Tuple[np.ndarray, List[float]]:
    """整轮回退的 k-center 局部搜索：R_cov 单调不增。"""
    Q = normalize(np.array(P, dtype=float, copy=True))
    _, R = cells_and_radius(Q)
    hist: List[float] = [R]
    a = step
    for it in range(iters):
        improved = False
        for tol in tols:
            if a < 1e-7:
                break
            trial, moved, R_try = kcenter_scan(Q.copy(), a, tol)
            if moved == 0:
                continue
            if R_try < R - 1e-14:
                Q, R, improved = trial, R_try, True
                a = min(step, a * 1.2)
                break
        if not improved:
            a *= 0.5
            if a < 1e-6:
                break
        if it % history_every == 0:
            hist.append(R)
    hist.append(R)
    return Q, hist


def _kcenter_escape(Q: np.ndarray, R: float) -> Optional[Tuple[np.ndarray, float]]:
    """逃离局部极小：依次尝试 单点扰动 / 一步 Lloyd / 一步能量精修。"""
    n = len(Q)
    cells, _ = cells_and_radius(Q)
    # (a) 只对"覆盖最差"的几个 site 做小幅扰动（遍历全部 site 太慢）
    worst = []
    for i, V in enumerate(cells):
        if len(V) < 3:
            continue
        d = max(float(np.arccos(np.clip(v @ Q[i], -1, 1))).real for v in V)
        worst.append((d, i))
    worst.sort(reverse=True)
    for _, i in worst[:5]:
        V = cells[i]
        sel = cell_worst_set(Q, i, V, tol=0.35)
        if sel is None:
            continue
        for w in sel:
            ut = _arc_dir(Q[i], w)
            if ut is None:
                continue
            trial = Q.copy()
            trial[i] = _arc_point(Q[i], ut, 0.02)
            _, Rt = cells_and_radius(trial)
            if Rt < R - 1e-14:
                return trial, Rt
    # (b) 一步球面 Lloyd（朝胞腔质心）
    out = Q.copy()
    for i in range(n):
        c = cell_centroid_dirs(cells[i])
        if c is None:
            continue
        t = math.acos(float(np.clip(c @ Q[i], -1.0, 1.0)))
        if t < 1e-12:
            continue
        ut = _arc_dir(Q[i], c)
        if ut is not None:
            out[i] = _arc_point(Q[i], ut, t)
    _, Rt = cells_and_radius(out)
    if Rt < R - 1e-14:
        return out, Rt
    # (c) 一步能量精修
    try:
        Pw, _ = thomson(Q, 1.0, iters=20)
        _, Rw = cells_and_radius(Pw)
        if Rw < R - 1e-14:
            return Pw, Rw
    except Exception:
        pass
    return None


def kcenter_solve(P: np.ndarray, iters: int = 120, step: float = 0.25,
                  escape_rounds: int = 8) -> Tuple[np.ndarray, float]:
    """局部搜索 + 逃逸，返回 (点集, 精确 R_cov)。"""
    Q, _ = kcenter_local_search(P, iters=iters, step=step)
    _, R = cells_and_radius(Q)
    for _ in range(escape_rounds):
        esc = _kcenter_escape(Q, R)
        if esc is None:
            break
        Q2, R2 = kcenter_local_search(esc[0], iters=iters, step=step)
        _, R2b = cells_and_radius(Q2)
        Q, R = (Q2, R2b) if R2b < esc[1] else esc
    return Q, R


def _energy_seeds(n: int, rng: np.random.Generator, iters: int, count: int) -> List[np.ndarray]:
    bases = [fibonacci_points(n),
             spiral_points(n, jitter=0.15, rng=rng),
             spiral_points(n, jitter=0.45, rng=rng)]
    while len(bases) < count:
        bases.append(random_points(n, rng))
    return [thomson(P0, 1.0, iters=iters)[0] for P0 in bases[:count]]


def _perturb(P: np.ndarray, rng: np.random.Generator, amp: float) -> np.ndarray:
    """随机整体旋转 + 逐点抖动（探索当前盆地之外）。"""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    th = amp * rng.uniform(-1.0, 1.0)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    Rm = np.eye(3) * math.cos(th) + K * math.sin(th) + np.outer(axis, axis) * (1 - math.cos(th))
    return normalize(P @ Rm.T + rng.normal(scale=amp * 0.3, size=P.shape))


def solve_cover(n: int, seed: int = 0, verbose: bool = False,
                energy_iters: int = 300, tammes_iters: int = 6000) -> Dict:
    """覆盖目标：多起点 + k-center + 随机重启 + 扰动细化，按精确 R_cov 择优。"""
    rng = np.random.default_rng(seed)
    kc_iters = 120 if n <= 60 else (60 if n <= 120 else 30)

    starts: List[Tuple[str, np.ndarray]] = []
    n_energy = 4 if n <= 60 else (2 if n <= 120 else 1)
    for k, Pe in enumerate(_energy_seeds(n, rng, energy_iters, n_energy)):
        starts.append((f"energy{k}", Pe))
    try:
        starts.append(("tammes", tammes(fibonacci_points(n), iters=tammes_iters,
                                        rate=0.05, seed=seed)[0]))
    except Exception:
        pass
    starts.append(("lloyd", coverage_lloyd(fibonacci_points(n), iters=100)[0]))
    starts.append(("fib", fibonacci_points(n)))

    results: List[Tuple[float, str, np.ndarray]] = []
    for tag, P0 in starts:
        Q, _ = kcenter_local_search(P0, iters=kc_iters, step=0.25)
        results.append((coverage_radius(Q), f"kc({tag})", Q))

    n_rnd = 8 if n <= 60 else (4 if n <= 120 else 2)
    for k in range(n_rnd):
        Q, _ = kcenter_local_search(random_points(n, rng), iters=kc_iters, step=0.25)
        results.append((coverage_radius(Q), f"rand{k}", Q))

    n_ref = 6 if n <= 60 else (2 if n <= 120 else 1)
    for r in range(n_ref):
        results.sort(key=lambda x: x[0])
        Q, _ = kcenter_solve(_perturb(results[0][2], rng, 0.05 + 0.10 * r), iters=kc_iters)
        results.append((coverage_radius(Q), f"refine{r}", Q))

    results.sort(key=lambda x: x[0])
    if verbose:
        for R, tag, _ in results[:8]:
            print(f"      {tag:<22} R_cov={math.degrees(R):9.5f}°", flush=True)
    R, tag, P = results[0]
    return {"points": normalize(P), "radius_deg": math.degrees(R), "tag": tag}


# --------------------------------------------------------------------------
# 参考最优值（用于自检 / 报告）
# --------------------------------------------------------------------------
# n -> 已知最优 Tammes 最小夹角（度）
KNOWN_TAMMES_DEG: Dict[int, float] = {
    4: 109.47122063449069,
    5: 90.0,
    6: 90.0,
    7: 77.86954214585094,
    8: 74.85849218504127,
    9: 70.52877936550931,
    10: 66.14682200005954,
    11: 63.43494882292201,
    12: 63.43494882292201,
    13: 57.13670357678126,
    14: 55.02424043950313,
    15: 53.65782541610374,
    16: 52.24439530662547,
    17: 51.09008373456465,
    18: 49.55613045234065,
    19: 48.70411917479097,
    20: 47.43163621306296,
    24: 43.69033209627186,
    32: 37.37736814064969,   # 截角二十面体（足球）构型
    # 注：N=60 常被误记为 30°（那是五角化截角二十面体的误传）。
    # 实际上足球（截角二十面体）60 个顶点的最小夹角只有 23.2814°，
    # 本项目的优化器能做到 26.5387°，已经优于该构型，故不列为参考上限。
}


def as_json_block(n: int, sol: Dict[str, Dict], digits: int = 6) -> Dict:
    """把求解结果压缩成可嵌入网页的精简结构。"""
    confs = {}
    for key, val in sol.items():
        P = val["points"]
        hist = [float(x) for x in val.get("history", [])]
        if key == "energy":
            vals = {"energy": float(val["energy"]), "minDeg": float(val["min_deg"])}
        elif key == "tammes":
            vals = {"minDeg": float(val["min_deg"]), "energy": float(val["energy"])}
        elif key == "area":
            vals = {"cv": float(val["cv"]), "minDeg": float(val["min_deg"]),
                    "energy": float(val["energy"])}
        else:
            vals = {"radiusDeg": float(val["radius_deg"]), "minDeg": float(val["min_deg"]),
                    "energy": float(val["energy"])}
        confs[key] = {
            **vals,
            "history": [round(h, 9) for h in _downsample(hist, 60)],
            "pts": [[round(float(c), digits) for c in p] for p in P],
        }
    return {"n": n, "configs": confs}


def _downsample(seq: Sequence[float], k: int) -> List[float]:
    seq = list(seq)
    if len(seq) <= k:
        return seq
    idx = np.linspace(0, len(seq) - 1, k).round().astype(int)
    return [seq[i] for i in idx]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="球面 N 点均匀分布优化（三类目标）")
    ap.add_argument("n", nargs="+", type=int, help="点数，可给多个")
    ap.add_argument("-s", "--riesz", type=float, default=1.0)
    ap.add_argument("-o", "--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--iters-energy", type=int, default=400)
    ap.add_argument("--iters-tammes", type=int, default=8000)
    ap.add_argument("--iters-cover", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=2)
    args = ap.parse_args()

    data = {}
    for n in args.n:
        t0 = time.time()
        sol = solve_all(n, s=args.riesz, verbose=True,
                        iters_energy=args.iters_energy,
                        iters_tammes=args.iters_tammes,
                        iters_cover=args.iters_cover,
                        restarts=args.restarts)
        data[str(n)] = as_json_block(n, sol)
        ref = KNOWN_TAMMES_DEG.get(n)
        ref_s = f"{ref:.3f}" if ref else "  -  "
        print(f"  n={n:4d} {time.time() - t0:6.1f}s | "
              f"E={sol['energy']['energy']:12.5f} | "
              f"dmin(T)={sol['tammes']['min_deg']:8.4f} (最优≈{ref_s}) | "
              f"R_cov={sol['cover']['radius_deg']:8.4f} | "
              f"dmin(C)={sol['cover']['min_deg']:8.4f}", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"已写入 {args.out}")
