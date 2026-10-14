"""数据集辅助工具（三个子命令）。

用法：
    python tools.py summary            # 打印 sphere_data.js 的质量汇总
    python tools.py refine 60 7 13     # 对指定 N 做多起点 Tammes + 能量抛光，更好则写回
    python tools.py reference          # 生成 tests/reference.json（前端交叉验证参考值）
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
from make_data import ENERGY_REF, summarize, write_out

OUT_JS = "sphere_data.js"


def load_data() -> Dict[str, Dict]:
    raw = open(OUT_JS, encoding="utf-8").read()
    return json.loads(raw[raw.index("=") + 1: raw.rindex(";")])


def refine_one(n: int, seed: int = 0) -> Dict:
    """多起点 Tammes（更长迭代）+ 能量抛光，返回最好的那组。"""
    rng = np.random.default_rng(seed)
    starts = [so.fibonacci_points(n),
              so.spiral_points(n, jitter=0.1, rng=rng),
              so.spiral_points(n, jitter=0.35, rng=rng),
              so.random_points(n, rng)]
    best = None
    for k, P0 in enumerate(starts):
        for rate in (0.05, 0.02):
            Pt, _ = so.tammes(P0, iters=30000, rate=rate, decay=0.9998, seed=seed + k)
            Pt2, _ = so.thomson(Pt, 1.0, iters=600)
            cand = Pt2 if so.min_angle(Pt2) >= so.min_angle(Pt) - 1e-12 else Pt
            d = so.min_angle(cand)
            if best is None or d > best[0]:
                best = (d, cand)
    d, P = best
    return {"points": P, "min_deg": math.degrees(d),
            "energy": so.riesz_energy(P, 1.0), "history": []}


def cmd_refine(ns: List[int]) -> None:
    data = load_data()
    for n in ns:
        key = str(n)
        old = data[key]["configs"]["tammes"]["minDeg"] if key in data else None
        t0 = time.time()
        res = refine_one(n)
        new = res["min_deg"]
        ref = so.KNOWN_TAMMES_DEG.get(n)
        mark = ""
        if old is None or new > old + 1e-9:
            data.setdefault(key, {"n": n, "configs": {}})
            data[key]["configs"]["tammes"] = {
                "minDeg": new, "energy": res["energy"], "history": [],
                "pts": [[round(float(c), 6) for c in p] for p in res["points"]],
            }
            mark = "已写回"
        print(f"n={n:4d} {time.time()-t0:6.1f}s | dmin "
              f"{(old if old else float('nan')):8.4f} -> {new:8.4f} | "
              f"最优 {(ref if ref else float('nan')):8.4f} | {mark}", flush=True)
    write_out(data)




# ---------------------------------------------------------------- reference
def make_reference(out_path: str = 'tests/reference.json') -> None:
    """生成前端交叉验证用的参考值（全部由 sphoptim.py 精确计算）。

    verify_frontend.js 会先看这个文件是否存在，不存在就调用本函数生成。
    """
    import numpy as np

    rng = np.random.default_rng(7)
    Ps = so.normalize(rng.normal(size=(7, 3)))
    riesz = {}
    for sv in (1.0, 0.0, 2.0, 3.0, 6.0, 12.0, 0.5, -1.0):
        riesz[str(sv)] = {
            's': sv,
            'energy': float(so.riesz_energy(Ps, sv)),
            'grad': so.energy_grad(Ps, sv).tolist(),
            'points': Ps.tolist(),
        }

    # 胞腔面积参考（前端 cellArea / areaCvOf 的对照）
    cell_area_ref: Dict[str, list] = {}
    cell_cv_ref: Dict[str, float] = {}

    configs: Dict[str, Dict] = {}
    if os.path.exists(OUT_JS):
        for k, v in load_data().items():
            row = {}
            for kk, vv in v['configs'].items():
                P = so.normalize(np.array(vv['pts'], float))
                row[kk] = {
                    'pts': [[float(c) for c in p] for p in P],
                    'energy': float(so.riesz_energy(P, 1.0)),
                    'minDeg': math.degrees(so.min_angle(P)),
                    'radiusDeg': math.degrees(so.coverage_radius(P)),
                }
            configs[k] = row
            if k in ('6', '12', '20', '50', '100'):
                Pc = so.normalize(np.array(row['cover']['pts'], float))
                Ar, _ = so.cell_areas(Pc)
                cell_area_ref[k] = [float(x) for x in Ar]
                cell_cv_ref[k] = float(so.area_uniformity(Pc)[0])

    ref = {
        'note': '由 tools.py reference 生成；所有数值来自 sphoptim.py 的精确实现',
        'polyhedra': {
            'tetra': {'minDeg': 109.47122063449069, 'radiusDeg': 70.52877936550931},
            'octa': {'minDeg': 90.0, 'radiusDeg': 54.73561031724535},
            'icosa': {'minDeg': 63.43494882292201, 'radiusDeg': 37.37736814064969},
        },
        'riesz_cases': riesz,
        'cell_area_ref': cell_area_ref,
        'cell_cv_ref': cell_cv_ref,
        'configs': configs,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(ref, f, ensure_ascii=False, separators=(",", ":"))
    print(f'{out_path} 已生成：{len(configs)} 组配置，{len(riesz)} 个 s 用例，'
          f'{os.path.getsize(out_path)/1024:.0f} KB')


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == 'refine':
        ns = [int(x) for x in sys.argv[2:]]
        if not ns:
            print('用法: python tools.py refine 60 7 13')
            return
        cmd_refine(ns)
    elif cmd == 'summary':
        summarize(load_data())
    elif cmd == 'reference':
        make_reference()
    else:
        print(__doc__)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == 'refine':
        ns = [int(x) for x in sys.argv[2:]]
        if not ns:
            print('用法: python tools.py refine 60 7 13')
            return
        cmd_refine(ns)
    elif cmd == 'summary':
        summarize(load_data())
    elif cmd == 'reference':
        make_reference()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
