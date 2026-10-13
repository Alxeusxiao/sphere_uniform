"""最终判定 s=0 的梯度：

(1) 在**扰动点 A** 处算闭式 2*Σ_j d^{-2}(p_i - p_j)（这才是有限差分真正在测的量）；
(2) 分别与 ±2*raw 对比，看哪一边吻合。
"""
import math
import numpy as np
import sphoptim as so

rng = np.random.default_rng(3)
P = so.normalize(rng.normal(size=(6, 3)))


def emove(p, q, t):
    c = float(np.clip(p @ q, -1, 1))
    sn = math.sqrt(max(1 - c * c, 0.0))
    if sn < 1e-14:
        return p
    return p * math.cos(t) + ((q - c * p) / sn) * math.sin(t)


def E_log_closed(Q):
    tot = 0.0
    for i in range(len(Q)):
        for j in range(i + 1, len(Q)):
            tot += math.log(max(float(((Q[i] - Q[j]) ** 2).sum()), 1e-300))
    return tot


def closed_grad_at(Q, i, factor):
    """在 Q 的 p_i 处：factor * Σ_j d^{-2}(p_i - p_j)"""
    g = np.zeros(3)
    for j in range(len(Q)):
        if j == i:
            continue
        d2 = max(float(((Q[i] - Q[j]) ** 2).sum()), 1e-300)
        g += (Q[i] - Q[j]) / d2
    return factor * g


def raw_at(Q, i):
    G = Q @ Q.T
    D2 = np.maximum(2.0 * (1.0 - G), 1e-30)
    np.fill_diagonal(D2, 1.0)
    W = (D2 ** -1.0).copy()
    np.fill_diagonal(W, 0.0)
    A = W @ Q
    c = float(A[i] @ Q[i])
    return A[i] - c * Q[i]


i = 0
q = rng.normal(size=3); q = q / np.linalg.norm(q)
q = q - P[i] * float(q @ P[i]); q = q / np.linalg.norm(q)
eps = 1e-6
Aa, Bb = P.copy(), P.copy()
Aa[i] = emove(P[i], q, eps); Bb[i] = emove(P[i], q, -eps)
fd = (E_log_closed(Aa) - E_log_closed(Bb)) / (2 * eps)

print("有限差分                         =", f"{fd:+.9f}")
for fac in (1.0, 2.0):
    print(f"  +{fac} * closed(在 A 处)      = {float(closed_grad_at(Aa, i, fac) @ q):+.9f}")
    print(f"  -{fac} * closed(在 A 处)      = {float(-closed_grad_at(Aa, i, fac) @ q):+.9f}")
r = raw_at(Aa, i)
print("  +2 * raw(A)                    =", f"{float(2*r@q):+.9f}")
print("  -2 * raw(A)                    =", f"{float(-2*r@q):+.9f}")
print()
print("=> 与有限差分同号的组合就是正确形式。")
