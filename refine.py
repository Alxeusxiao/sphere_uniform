"""对指定 N 做"高强度 Tammes 重算"：更多步数 + 多次随机重启 + 能量抛光，
若 dmin 有改善则写回 sphere_data.json / sphere_data.js。

用法： python refine.py 60 7 13 11 34
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np

import sphoptim as so

DATA = "sphere_data.json"


def refine(n: int, seed: int = 0) -> Dict:
    """多起点 Tammes + 能量抛光，返回最好的一组结果。"""
    rng = np.random.default_rng(seed)
    starts = [so.fibonacci_points(n),
              so.spiral_points(n, jitter=0.1, rng=rng),
              so.spiral_points(n, jitter=0.35, rng=rng),
              so.random_points(n, rng)]
    best = None
    for k, P0 in enumerate(starts):
        for rate in (0.05, 0.02):
            Pt, _ = so.tammes(P0, iters=30000, rate=rate, decay=0.9998, seed=seed + k)
            # 能量抛光
            Pt2, _ = so.thomson(Pt, 1.0, iters=600)
            cand = Pt2 if so.min_angle(Pt2) >= so.min_angle(Pt) - 1e-12 else Pt
            d = so.min_angle(cand)
            if best is None or d > best[0]:
                best = (d, cand)
    d, P = best
    return {
        "points": P,
        "min_deg": math.degrees(d),
        "energy": so.riesz_energy(P, 1.0),
        "history": [],
    }


def main() -> None:
    ns: List[int] = [int(x) for x in sys.argv[1:]]
    if not ns:
        print("用法: python refine.py 60 7 13 ...")
        return
    data = json.load(open(DATA, encoding="utf-8"))
    for n in ns:
        key = str(n)
        old = data[key]["configs"]["tammes"]["minDeg"] if key in data else None
        t0 = time.time()
        res = refine(n)
        new = res["min_deg"]
        ref = so.KNOWN_TAMMES_DEG.get(n)
        mark = ""
        if old is None or new > old + 1e-9:
            data.setdefault(key, {"n": n, "configs": {}})
            data[key]["configs"]["tammes"] = {
                "minDeg": new,
                "energy": res["energy"],
                "history": [],
                "pts": [[round(float(c), 6) for c in p] for p in res["points"]],
            }
            mark = "已写回"
        print(f"n={n:4d} {time.time()-t0:6.1f}s | dmin {old if old else float('nan'):8.4f} "
              f"-> {new:8.4f} | 最优 {ref if ref else float('nan'):8.4f} | {mark}", flush=True)

    with open(DATA, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    with open("sphere_data.js", "w", encoding="utf-8") as f:
        f.write("window.SPHERE_DATA=")
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    print(f"\n{len(data)} 组已写回，sphere_data.json {os.path.getsize(DATA)/1024:.0f} KB")


if __name__ == "__main__":
    main()
