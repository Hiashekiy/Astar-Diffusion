"""Benchmark the four propagation backends at 64/128/256 on three map complexities.

    python scripts/benchmark_propagation.py
    python scripts/benchmark_propagation.py --sizes 64,128 --repeats 5

Timing and memory are measured in separate calls on purpose: ``tracemalloc`` intercepts every
allocation and inflates wall time by more than an order of magnitude, unevenly across
implementations, which would make the comparison meaningless.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_field.grid import Grid  # noqa: E402
from guidance_field.paths import astar  # noqa: E402
from guidance_field.propagation import (  # noqa: E402
    propagate_bucket,
    propagate_heap,
    propagate_scipy,
    propagate_vectorized,
)

LAM = 0.02
NUM_SOURCES = 64


def make_clutter(size, rng):
    occ = np.zeros((size, size), dtype=np.int8)
    target = 0.22 * size * size
    while occ.sum() < target:
        h = int(rng.integers(2, max(3, size // 10)))
        w = int(rng.integers(2, max(3, size // 8)))
        r = int(rng.integers(1, size - h - 1))
        c = int(rng.integers(1, size - w - 1))
        occ[r:r + h, c:c + w] = 1
    return occ


def make_maze(size, rng):
    occ = np.ones((size, size), dtype=np.int8)
    n = (size - 1) // 2
    for i in range(n):
        for j in range(n):
            occ[2 * i + 1, 2 * j + 1] = 0
    visited = np.zeros((n, n), dtype=bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        i, j = stack[-1]
        cand = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= i + di < n and 0 <= j + dj < n and not visited[i + di, j + dj]]
        if not cand:
            stack.pop()
            continue
        ni, nj = cand[int(rng.integers(len(cand)))]
        visited[ni, nj] = True
        occ[i + ni + 1, j + nj + 1] = 0
        stack.append((ni, nj))
    return occ


def make_rooms(size, rng):
    occ = np.ones((size, size), dtype=np.int8)
    step = max(8, size // 5)
    for r in range(2, size - 2, step):
        for c in range(2, size - 2, step):
            occ[r:min(size - 1, r + step - 2), c:min(size - 1, c + step - 2)] = 0
    for r in range(2, size - 2, step):
        for c in range(2, size - 2, step):
            occ[min(size - 2, r + step - 2):min(size - 1, r + step - 1), c:c + 3] = 0
            occ[r:r + 3, min(size - 2, c + step - 2):min(size - 1, c + step - 1)] = 0
    return occ


MAP_KINDS = {"clutter": make_clutter, "maze": make_maze, "rooms": make_rooms}


def source_set(grid, rng):
    free = np.argwhere(grid.free)
    a = tuple(int(v) for v in free[int(rng.integers(len(free)))])
    dist = grid.geodesic_distance([grid.idx(*a)])
    b = grid.rc(int(np.argmax(np.where(dist > 0, dist, -1))))
    nodes = astar(grid, grid.idx(*a), grid.idx(*b))
    pts = np.array([grid.rc(int(i)) for i in nodes], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    sel = np.linspace(0, cum[-1], NUM_SOURCES)
    samples = np.rint(np.stack([[np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])]
                                for t in sel])).astype(int)
    return samples, np.linspace(0.0, 1.0, NUM_SOURCES)


def timeit(fn, repeats):
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0, out


def peak_mib(fn):
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="64,128,256")
    ap.add_argument("--kinds", default="clutter,maze,rooms")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outdir", default="outputs/benchmark")
    args = ap.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(",")]
    kinds = [k.strip() for k in args.kinds.split(",")]
    out = Path(args.outdir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    backends = (("heap", propagate_heap), ("vectorized", propagate_vectorized),
                ("bucket", propagate_bucket), ("scipy", propagate_scipy))
    rows = []
    print(f"{'size':>5} {'kind':8s} " + " ".join(f"{n:>11s}" for n, _ in backends) + "   max|dF| scipy")
    for size in sizes:
        for kind in kinds:
            rng = np.random.default_rng(hash((size, kind)) % (2 ** 31))
            occ = MAP_KINDS[kind](size, rng)
            occ[0, :] = occ[-1, :] = 1
            occ[:, 0] = occ[:, -1] = 1
            grid = Grid(occ)
            samples, weights = source_set(grid, rng)
            reps = max(1, args.repeats if size <= 128 else 2)
            reference = None
            cells = []
            for name, fn in backends:
                ms, result = timeit(lambda f=fn: f(grid, samples, weights, LAM), reps)
                if name == "heap":
                    reference = result
                rows.append({"size": size, "kind": kind, "method": name, "ms": round(ms, 4),
                             "peak_mib": round(peak_mib(lambda f=fn: f(grid, samples, weights, LAM)), 4),
                             "free_cells": int(grid.free.sum())})
                cells.append(f"{ms:11.2f}")
            scipy_res = propagate_scipy(grid, samples, weights, LAM)
            ok = grid.free & np.isfinite(reference) & np.isfinite(scipy_res)
            err = float(np.max(np.abs(reference[ok] - scipy_res[ok]))) if ok.any() else 0.0
            print(f"{size:5d} {kind:8s} " + " ".join(cells) + f"   {err:.1e}")

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["size", "kind", "method", "ms", "peak_mib", "free_cells"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps({"lam": LAM, "num_sources": NUM_SOURCES,
                                                  "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n-> {out / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
