"""生成网页用的完整数据。

沙箱环境不允许命名管道（multiprocessing.Pool 会失败），所以并行方式是：
父进程把 N 按预计耗时切成若干块，用 subprocess 启动多个工人进程，每个工人把
结果写到自己的临时 JSON 文件；全部结束后父进程合并。

用法：
    python make_data_par.py                    # 默认 N 列表，自动选进程数
    python make_data_par.py 7 13 30
    python make_data_par.py --workers 8
    python make_data_par.py --worker 1 8 6 12  # 内部用：第 1/8 个工人
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List

import sphoptim as so

DEFAULT_N = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26,
             28, 30, 32, 34, 36, 40, 44, 48, 50, 56, 60, 64, 72, 80, 90, 100,
             110, 120, 130, 150, 170, 200]


def cover_iters(n: int) -> int:
    """Lloyd 轮数随 N 自适应：大 N 每轮贵，减少轮数（收敛本来也更快）。"""
    if n <= 40:
        return 200
    if n <= 80:
        return 140
    if n <= 120:
        return 90
    if n <= 160:
        return 60
    return 45


def energy_iters(n: int) -> int:
    """能量 FW 轮数：每轮要一次 O(n^2) 一维搜索，大 N 收敛也更快。"""
    if n <= 40:
        return 400
    if n <= 100:
        return 250
    if n <= 160:
        return 150
    return 80


def tammes_iters(n: int) -> int:
    return 8000 if n <= 40 else (4000 if n <= 100 else 2500)


def run_chunk(ns: List[int], tag: str) -> None:
    """工人：求解自己负责的 N，结果写入 chunk_<tag>.json。"""
    out: Dict[str, Dict] = {}
    for n in ns:
        t0 = time.time()
        sol = so.solve_all(n, s=1.0, verbose=False, iters_energy=energy_iters(n),
                           iters_tammes=tammes_iters(n),
                           iters_cover=cover_iters(n), restarts=2)
        out[str(n)] = so.as_json_block(n, sol)
        c = out[str(n)]["configs"]
        ref = so.KNOWN_TAMMES_DEG.get(n)
        print(f"[{tag}] n={n:4d} {time.time()-t0:6.1f}s | "
              f"E={c['energy']['energy']:12.5f} | "
              f"dmin_T={c['tammes']['minDeg']:8.4f}"
              f"{' (最优≈%.4f)' % ref if ref else ''} | "
              f"R_cov={c['cover']['radiusDeg']:8.4f}", flush=True)
    with open(f"chunk_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


def main() -> None:
    argv = sys.argv[1:]

    # ---- 工人模式（必须先剥掉 --worker，再解析 N 列表） ----
    if "--worker" in argv:
        k = argv.index("--worker")
        idx, total = int(argv[k + 1]), int(argv[k + 2])
        rest = argv[:k] + argv[k + 3:]
        ns = [int(x) for x in rest if x.lstrip("-").isdigit()]
        # 父进程已经把分片算好并整个交给本工人，这里直接求解
        run_chunk(ns, f"{idx}of{total}")
        return

    workers = max(1, (os.cpu_count() or 4) - 2)
    if "--workers" in argv:
        k = argv.index("--workers")
        workers = int(argv[k + 1])
        del argv[k:k + 2]
    ns: List[int] = [int(x) for x in argv] or DEFAULT_N

    # 按预计耗时做均衡划分（大 N 单独成块）
    order = sorted(ns, key=lambda x: -x)
    chunks: List[List[int]] = [[] for _ in range(workers)]
    load = [0.0] * workers
    for n in order:
        k = load.index(min(load))
        chunks[k].append(n)
        load[k] += n ** 3
    for c in chunks:
        c.sort()

    print(f"并行求解 {len(ns)} 组，{workers} 个工人进程", flush=True)
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
        lf.flush()
        lf.close()
        if rc != 0:
            failed.append((idx, rc, c))
            print(f"  工人{idx} 退出码 {rc}，负责 {c}", flush=True)

    data: Dict[str, Dict] = {}
    for idx in range(workers):
        f = f"chunk_{idx}of{workers}.json"
        if os.path.exists(f):
            try:
                data.update(json.load(open(f, encoding="utf-8")))
            except Exception as e:
                print(f"  读取 {f} 失败: {e!r}", flush=True)
            os.remove(f)
    missing = [n for n in ns if str(n) not in data]
    if missing:
        print("缺失:", missing, flush=True)
        for idx, rc, c in failed:
            print(f"--- 工人{idx}.log 末尾 ---", flush=True)
            try:
                lines = open(f"worker_{idx}.log", encoding="utf-8").read().strip().splitlines()
                print("\n".join(lines[-20:]), flush=True)
            except Exception:
                pass
        return

    with open("sphere_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    with open("sphere_data.js", "w", encoding="utf-8") as f:
        f.write("window.SPHERE_DATA=")
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    for idx in range(workers):
        try:
            os.remove(f"worker_{idx}.log")
        except OSError:
            pass
    size = os.path.getsize("sphere_data.json")
    print(f"\n完成 {len(data)} 组，总用时 {time.time()-t_start:.1f}s，"
          f"sphere_data.json {size/1024:.0f} KB", flush=True)


if __name__ == "__main__":
    main()
