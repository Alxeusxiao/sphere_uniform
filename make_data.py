"""生成网页用的数据集（多进程 + 断点续跑 + 合并 + 汇总，一个文件搞定）。

沙箱不允许命名管道（multiprocessing.Pool 会 PermissionError），所以并行方式是：
父进程按预计耗时把 N 分块，用 subprocess 起"工人"进程，每个工人把结果写到
自己的 chunk_<i>of<N>.json；全部结束后父进程合并成 sphere_data.js。

用法：
    python make_data.py                     # 默认 N 列表，自动选进程数
    python make_data.py 7 13 30             # 只算指定的 N
    python make_data.py --workers 12
    python make_data.py --merge             # 只合并已有分片 + 补算缺失
    python make_data.py --summary           # 只打印已有数据的质量汇总
"""
from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np

import sphoptim as so

OUT_JS = "sphere_data.js"
DEFAULT_N = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26,
             28, 30, 32, 34, 36, 40, 44, 48, 50, 56, 60, 64, 72, 80, 90, 100,
             110, 120, 130, 150, 170, 200]

# 文献最优值（用于对照；能对照的 N 有限）
ENERGY_REF = {4: 3.6742346, 5: 6.4744091, 6: 9.9852814, 7: 14.4522529, 8: 19.6752879,
              9: 25.7599865, 10: 32.7169495, 11: 40.5964505, 12: 49.1652531,
              13: 58.8532306, 14: 69.3063633, 15: 80.6702441, 16: 92.9116553,
              18: 120.0844674, 20: 150.8815683}


def cover_iters(n: int) -> int:
    return 200 if n <= 40 else (140 if n <= 80 else (90 if n <= 120 else (60 if n <= 160 else 45)))


def energy_iters(n: int) -> int:
    return 400 if n <= 40 else (250 if n <= 100 else (150 if n <= 160 else 80))


def tammes_iters(n: int) -> int:
    return 8000 if n <= 40 else (4000 if n <= 100 else 2500)


def solve_one(n: int) -> Dict:
    sol = so.solve_all(n, s=1.0, verbose=False, iters_energy=energy_iters(n),
                       iters_tammes=tammes_iters(n), iters_cover=cover_iters(n),
                       restarts=2)
    return so.as_json_block(n, sol)


def _fmt_line(n: int, block: Dict, dt: float) -> str:
    c = block["configs"]
    ref = so.KNOWN_TAMMES_DEG.get(n)
    ref_s = f"{ref:8.4f}" if ref else "    -   "
    return (f"n={n:4d} {dt:6.1f}s | E={c['energy']['energy']:12.5f} | "
            f"dmin_T={c['tammes']['minDeg']:8.4f} (最优≈{ref_s}) | "
            f"R_cov={c['cover']['radiusDeg']:8.4f} | dmin_C={c['cover']['minDeg']:7.4f}")


# ---------------------------------------------------------------- 工人
def run_chunk(ns: List[int], tag: str) -> None:
    out: Dict[str, Dict] = {}
    for n in ns:
        t0 = time.time()
        out[str(n)] = solve_one(n)
        print(f"[{tag}] " + _fmt_line(n, out[str(n)], time.time() - t0), flush=True)
    with open(f"chunk_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


# ---------------------------------------------------------------- 汇总 / 写出
def summarize(data: Dict[str, Dict]) -> None:
    keys = sorted(int(k) for k in data)
    print(f"\n{'N':>4} {'能量':>13} {'最优能量':>13} {'偏差':>8} "
          f"{'dmin(T)':>9} {'最优dmin':>9} {'达成':>7} {'R_cov':>9} "
          f"{'cover/下界':>10} {'面积cv':>9}")
    e_bad, d_ok = [], []
    for n in keys:
        c = data[str(n)]["configs"]
        E, dT, R = c["energy"]["energy"], c["tammes"]["minDeg"], c["cover"]["radiusDeg"]
        CV = c["area"]["cv"] if "area" in c else float("nan")
        e_ref, d_ref = ENERGY_REF.get(n), so.KNOWN_TAMMES_DEG.get(n)
        lb = math.degrees(math.acos(max(-1.0, 1 - 2.0 / n)))
        if e_ref:
            e_bad.append(abs(E - e_ref) / e_ref)
        if d_ref:
            d_ok.append(dT / d_ref)
        print(f"{n:>4} {E:13.4f} {(f'{e_ref:13.4f}' if e_ref else f'{"-":>13}')} "
              f"{(f'{100*(E-e_ref)/e_ref:+7.3f}%' if e_ref else f'{"-":>8}')} "
              f"{dT:9.4f} {(f'{d_ref:9.4f}' if d_ref else f'{"-":>9}')} "
              f"{(f'{100*dT/d_ref:6.1f}%' if d_ref else f'{"-":>7}')} "
              f"{R:9.4f} {R/lb:10.4f} {CV:9.5f}")
    if e_bad:
        print(f"\n能量：{len(e_bad)} 组有文献最优值，最大偏差 {100*max(e_bad):.4f}%，"
              f"平均 {100*sum(e_bad)/len(e_bad):.4f}%")
    if d_ok:
        print(f"Tammes：{len(d_ok)} 组有最优值，最差达成 {100*min(d_ok):.1f}%，"
              f"平均 {100*sum(d_ok)/len(d_ok):.1f}%")


def write_out(data: Dict[str, Dict]) -> None:
    with open(OUT_JS, "w", encoding="utf-8") as f:
        f.write("window.SPHERE_DATA=")
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    print(f"\n写出 {OUT_JS}：{len(data)} 组，{os.path.getsize(OUT_JS)/1024:.0f} KB", flush=True)


def load_chunks() -> Dict[str, Dict]:
    data: Dict[str, Dict] = {}
    for f in sorted(glob.glob("chunk_*.json")):
        try:
            part = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"  跳过 {f}: {e!r}")
            continue
        if part:
            print(f"  读入 {f}: {len(part)} 组")
            data.update(part)
    return data


# ---------------------------------------------------------------- 主流程
def main() -> None:
    argv = sys.argv[1:]

    if "--worker" in argv:                      # 工人模式
        k = argv.index("--worker")
        idx, total = int(argv[k + 1]), int(argv[k + 2])
        rest = argv[:k] + argv[k + 3:]
        ns = [int(x) for x in rest if x.lstrip("-").isdigit()]
        run_chunk(ns, f"{idx}of{total}")
        return

    if "--summary" in argv:                     # 只打印
        if not os.path.exists(OUT_JS):
            print(f"{OUT_JS} 不存在")
            return
        raw = open(OUT_JS, encoding="utf-8").read()
        data = json.loads(raw[raw.index("=") + 1: raw.rindex(";")])
        summarize(data)
        return

    workers = max(1, (os.cpu_count() or 4) - 2)
    if "--workers" in argv:
        k = argv.index("--workers")
        workers = int(argv[k + 1])
        del argv[k:k + 2]
    only_merge = "--merge" in argv
    if only_merge:
        argv.remove("--merge")
        ns: List[int] = [int(x) for x in argv] or DEFAULT_N
    else:
        ns = [int(x) for x in argv] or DEFAULT_N

    have = load_chunks() if only_merge else {}
    todo = [n for n in ns if str(n) not in have] if only_merge else ns

    if todo:
        print(f"并行求解 {len(todo)} 组，{workers} 个工人进程", flush=True)
        order = sorted(todo, key=lambda x: -x)
        chunks: List[List[int]] = [[] for _ in range(workers)]
        load = [0.0] * workers
        for n in order:
            k = load.index(min(load))
            chunks[k].append(n)
            load[k] += n ** 3
        for c in chunks:
            c.sort()
        for i, c in enumerate(chunks):
            if c:
                print(f"  工人{i}: {c}", flush=True)

        py = sys.executable
        t_start = time.time()
        procs = []
        for idx, c in enumerate(chunks):
            if not c:
                continue
            lf = open(f"worker_{idx}.log", "w", encoding="utf-8")
            p = subprocess.Popen(
                [py, "-u", os.path.basename(__file__), "--worker", str(idx), str(workers)]
                + [str(x) for x in c],
                stdout=lf, stderr=subprocess.STDOUT, cwd=os.getcwd())
            procs.append((idx, c, p, lf))
        failed = []
        for idx, c, p, lf in procs:
            rc = p.wait()
            lf.close()
            if rc != 0:
                failed.append(idx)
                print(f"  工人{idx} 退出码 {rc}，负责 {c}", flush=True)

        for idx in range(workers):
            f = f"chunk_{idx}of{workers}.json"
            if os.path.exists(f):
                have.update(json.load(open(f, encoding="utf-8")))
                os.remove(f)
        missing = [n for n in ns if str(n) not in have]
        if missing:
            print("缺失:", missing, flush=True)
            for idx in failed:
                print(f"--- 工人{idx}.log 末尾 ---", flush=True)
                try:
                    lines = open(f"worker_{idx}.log", encoding="utf-8").read().strip().splitlines()
                    print("\n".join(lines[-20:]), flush=True)
                except Exception:
                    pass
            return
        print(f"\n求解完成，用时 {time.time()-t_start:.1f}s", flush=True)
        for idx in range(workers):
            try:
                os.remove(f"worker_{idx}.log")
            except OSError:
                pass

    summarize(have)
    write_out(have)


if __name__ == "__main__":
    main()
