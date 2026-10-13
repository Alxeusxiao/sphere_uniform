"""合并已有的 chunk_*.json，并（可选）补算缺失的 N，最后写出
sphere_data.json / sphere_data.js。

用法：
    python merge_chunks.py              # 只合并现有 chunk
    python merge_chunks.py 200 250      # 先补算这些 N，再合并
    python merge_chunks.py --list       # 只列出已有/缺失
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List

import sphoptim as so
from make_data_par import cover_iters, energy_iters, tammes_iters

OUT_JSON = "sphere_data.json"
OUT_JS = "sphere_data.js"


def load_chunks() -> Dict[str, Dict]:
    data: Dict[str, Dict] = {}
    for f in sorted(glob.glob("chunk_*.json")):
        try:
            part = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"跳过 {f}: {e!r}")
            continue
        if part:
            data.update(part)
        print(f"  读入 {f}: {len(part)} 组")
    return data


def compute(ns: List[int]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for n in ns:
        t0 = __import__("time").time()
        sol = so.solve_all(n, s=1.0, verbose=False, iters_energy=energy_iters(n),
                           iters_tammes=tammes_iters(n),
                           iters_cover=cover_iters(n), restarts=2)
        out[str(n)] = so.as_json_block(n, sol)
        c = out[str(n)]["configs"]
        ref = so.KNOWN_TAMMES_DEG.get(n)
        print(f"  补算 n={n:4d} {__import__('time').time()-t0:6.1f}s | "
              f"E={c['energy']['energy']:12.5f} | dmin_T={c['tammes']['minDeg']:8.4f}"
              f"{' (最优≈%.4f)' % ref if ref else ''} | "
              f"R_cov={c['cover']['radiusDeg']:8.4f}", flush=True)
    return out


def write(data: Dict[str, Dict]) -> None:
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    with open(OUT_JS, "w", encoding="utf-8") as f:
        f.write("window.SPHERE_DATA=")
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    keys = sorted(int(k) for k in data)
    print(f"\n写出 {len(data)} 组: {keys}")
    print(f"{OUT_JSON} {os.path.getsize(OUT_JSON)/1024:.0f} KB, "
          f"{OUT_JS} {os.path.getsize(OUT_JS)/1024:.0f} KB")


def main() -> None:
    args = sys.argv[1:]
    print("现有分片:")
    data = load_chunks()
    if args and args[0] == "--list":
        print("已有 N:", sorted(int(k) for k in data))
        return
    if args:
        data.update(compute([int(x) for x in args]))
    write(data)


if __name__ == "__main__":
    main()
