"""各 s 下 Python 能量与梯度的自洽性检验（中心差分 + 指数映射）。

注意：方向向量必须归一化，否则 (A[i]-P[i])/eps 不是单位切向量，
有限差分与解析梯度就不可比（这是之前那个 FAIL 的原因）。
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
    ut = (q - c * p) / sn
    return p * np.cos(t) + ut * np.sin(t)


ok = True
for s in (1.0, 0.0, 0.5, 3.0, 6.0, -1.0):
    G = so.energy_grad(P, s)
    worst = 0.0
    for i in range(6):
        q = rng.normal(size=3)
        q = q / np.linalg.norm(q)          # 关键：单位方向
        eps = 1e-6
        A, B = P.copy(), P.copy()
        A[i] = emove(P[i], q, eps)
        B[i] = emove(P[i], q, -eps)
        fd = (so.riesz_energy(A, s) - so.riesz_energy(B, s)) / (2 * eps)
        ut = (A[i] - P[i]) / eps
        worst = max(worst, abs(fd - float(G[i] @ ut)) / max(1e-12, abs(fd)))
    good = worst < 1e-6
    ok &= good
    print(f"  s={s:5.1f}  E={so.riesz_energy(P, s):14.6f}  能量-梯度最大相对误差={worst:.2e}  "
          f"{'OK' if good else 'FAIL'}")
print("各 s 下能量与梯度自洽 ✓" if ok else "有 FAIL")
