"""Fair benchmark of the three multi-source propagators (plan: "先做三种实现的公平 Benchmark").

    python scripts/benchmark_propagation.py                    # 64/128/256, 3 map kinds
    python scripts/benchmark_propagation.py --sizes 64,128     # subset
    python scripts/benchmark_propagation.py --repeats 5

Records for every (size, map kind, method): build time (best of N), max |dF| against the
heap implementation, peak memory, and the owner-handling / component-restriction variants.
Writes `outputs/benchmark_propagation/{results.csv,summary.json,benchmark.png}`.
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

from guidance_stage1.grid import Grid  # noqa: E402
from guidance_stage2.propagation import (  # noqa: E402
    propagate_auto,
    propagate_bucket,
    propagate_heap,
    propagate_vectorized,
)

LAM = 0.02
NUM_SOURCES = 64


# --------------------------------------------------------------------------- #
# benchmark maps: cheap constructions at any size, three complexity regimes
# --------------------------------------------------------------------------- #
def make_clutter(size: int, rng: np.random.Generator) -> np.ndarray:
    occ = np.zeros((size, size), dtype=np.int8)
    target = 0.22 * size * size
    while occ.sum() < target:
        h = int(rng.integers(2, max(3, size // 10)))
        w = int(rng.integers(2, max(3, size // 8)))
        r = int(rng.integers(1, size - h - 1))
        c = int(rng.integers(1, size - w - 1))
        occ[r : r + h, c : c + w] = 1
    occ[0, :] = occ[-1, :] = 1
    occ[:, 0] = occ[:, -1] = 1
    return occ


def make_maze(size: int, rng: np.random.Generator) -> np.ndarray:
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
    occ[0, :] = occ[-1, :] = 1
    occ[:, 0] = occ[:, -1] = 1
    return occ


def make_rooms(size: int, rng: np.random.Generator) -> np.ndarray:
    occ = np.ones((size, size), dtype=np.int8)
    step = max(8, size // 5)
    for r in range(2, size - 2, step):
        for c in range(2, size - 2, step):
            occ[r : min(size - 1, r + step - 2), c : min(size - 1, c + step - 2)] = 0
    for r in range(2, size - 2, step):  # doors between rooms
        for c in range(2, size - 2, step):
            occ[min(size - 2, r + step - 2) : min(size - 1, r + step - 1), c : c + 3] = 0
            occ[r : r + 3, min(size - 2, c + step - 2) : min(size - 1, c + step - 1)] = 0
    occ[0, :] = occ[-1, :] = 1
    occ[:, 0] = occ[:, -1] = 1
    return occ


MAP_KINDS = {"clutter": make_clutter, "maze": make_maze, "rooms": make_rooms}


def source_set(grid: Grid, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """64 sources spread along a real shortest path between two far free cells."""
    free = np.argwhere(grid.free)
    if len(free) < 10:
        raise RuntimeError("map has almost no free space")
    a = tuple(int(v) for v in free[int(rng.integers(len(free)))])
    dist = grid.geodesic_distance([grid.idx(*a)])
    far = int(np.argmax(np.where(dist > 0, dist, -1)))
    b = grid.rc(far)
    from guidance_stage1.paths import astar

    nodes = astar(grid, grid.idx(*a), grid.idx(*b))
    pts = np.array([grid.rc(int(i)) for i in nodes], dtype=float)
    if len(pts) < 2:
        raise RuntimeError("no path for the benchmark source set")
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    sel = np.linspace(0, cum[-1], NUM_SOURCES)
    samples = np.rint(np.stack([[np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])] for t in sel])).astype(int)
    weights = np.linspace(0.0, 1.0, NUM_SOURCES)
    return samples, weights


def timeit(fn, repeats: int) -> tuple[float, float, float, object]:
    """best / median wall time in ms, and peak MiB allocated during a separate call.

    Timing and memory are measured in *separate* calls on purpose: tracemalloc intercepts
    every allocation and inflates wall time by more than an order of magnitude, which would
    make the comparison meaningless.
    """
    best = float("inf")
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
        best = min(best, dt)
    tracemalloc.start()
    fn()
    _, peak_now = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return best, float(np.median(times)), peak_now / (1024 * 1024), out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="64,128,256")
    ap.add_argument("--kinds", default="clutter,maze,rooms")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outdir", default="outputs/benchmark_propagation")
    args = ap.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(",")]
    kinds = [k.strip() for k in args.kinds.split(",")]
    out = Path(args.outdir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for size in sizes:
        for kind in kinds:
            rng = np.random.default_rng(hash((size, kind)) % (2 ** 31))
            occ = MAP_KINDS[kind](size, rng)
            grid = Grid(occ)
            samples, weights = source_set(grid, rng)
            reps = max(1, args.repeats if size <= 128 else 2)

            ref_best, _, ref_peak, ref = timeit(
                lambda: propagate_heap(grid, samples, weights, LAM), reps)
            rows.append(dict(size=size, kind=kind, method="heap", ms_best=ref_best, ms_median=ref_best,
                             peak_mib=ref_peak, max_abs_err=0.0, speedup=1.0,
                             free_cells=int(grid.free.sum())))
            results = {"heap": ref}
            for name, fn in (("vectorized", propagate_vectorized), ("bucket", propagate_bucket)):
                best, med, peak, res = timeit(lambda f=fn: f(grid, samples, weights, LAM), reps)
                ok = grid.free & np.isfinite(ref) & np.isfinite(res)
                err = float(np.max(np.abs(ref[ok] - res[ok]))) if ok.any() else float("nan")
                nan_mismatch = int((np.isnan(ref) != np.isnan(res)).sum())
                rows.append(dict(size=size, kind=kind, method=name, ms_best=best, ms_median=med,
                                 peak_mib=peak, max_abs_err=err, speedup=ref_best / best if best else np.nan,
                                 free_cells=int(grid.free.sum()), nan_mismatch=nan_mismatch))
                results[name] = res
            print(f"[{size:3d} {kind:7s}] heap {ref_best:8.2f} ms | "
                  f"vectorized {rows[-2]['ms_best']:8.2f} ms ({rows[-2]['speedup']:4.2f}x, "
                  f"err {rows[-2]['max_abs_err']:.1e}) | "
                  f"bucket {rows[-1]['ms_best']:8.2f} ms ({rows[-1]['speedup']:4.2f}x, "
                  f"err {rows[-1]['max_abs_err']:.1e})", flush=True)

            # micro-optimisation: owner bookkeeping costs about 15% of the heap run, and
            # it is only needed for visualisation, never by the searcher
            no_owner, _, _, _ = timeit(
                lambda: propagate_heap(grid, samples, weights, LAM, return_owner=False), reps)
            with_owner, _, _, _ = timeit(
                lambda: propagate_heap(grid, samples, weights, LAM, return_owner=True), reps)
            full_comp, _, _, _ = timeit(lambda: propagate_auto(grid, samples, weights, LAM), reps)
            rows.append(dict(size=size, kind=kind, method="heap_no_owner", ms_best=no_owner,
                             ms_median=no_owner, peak_mib=0.0, max_abs_err=0.0,
                             speedup=ref_best / no_owner if no_owner else np.nan,
                             free_cells=int(grid.free.sum())))
            rows.append(dict(size=size, kind=kind, method="heap_with_owner", ms_best=with_owner,
                             ms_median=with_owner, peak_mib=0.0, max_abs_err=0.0,
                             speedup=ref_best / with_owner if with_owner else np.nan,
                             free_cells=int(grid.free.sum())))
            rows.append(dict(size=size, kind=kind, method="auto",
                             ms_best=full_comp, ms_median=full_comp, peak_mib=0.0, max_abs_err=0.0,
                             speedup=ref_best / full_comp if full_comp else np.nan,
                             free_cells=int(grid.free.sum())))

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})

    summary = {
        "params": {"sizes": sizes, "kinds": kinds, "repeats": args.repeats, "lam": LAM,
                   "num_sources": NUM_SOURCES},
        "rows": rows,
        "worst_error": {
            m: max([r["max_abs_err"] for r in rows if r["method"] == m], default=0.0)
            for m in ("heap", "vectorized", "bucket")
        },
        "best_speedup": {
            m: max([r["speedup"] for r in rows if r["method"] == m and r["speedup"] == r["speedup"]],
                   default=float("nan"))
            for m in ("vectorized", "bucket")
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), dpi=110)
    colors = {"heap": "#333333", "vectorized": "#1f77b4", "bucket": "#2ca02c"}
    for ax, kind in zip(axes, kinds):
        for method in ("heap", "vectorized", "bucket"):
            xs, ys = [], []
            for size in sizes:
                sel = [r for r in rows if r["size"] == size and r["kind"] == kind and r["method"] == method]
                if sel:
                    xs.append(size)
                    ys.append(sel[0]["ms_best"])
            ax.plot(xs, ys, "o-", color=colors[method], label=method, lw=1.8)
        ax.set_title(f"map kind: {kind}", fontsize=9.5)
        ax.set_xlabel("grid side")
        ax.set_ylabel("best build time [ms]")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle("Multi-source propagation benchmark (64 sources, lambda=0.02)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "benchmark.png", facecolor="white")
    plt.close(fig)
    print(f"\n-> {out / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
