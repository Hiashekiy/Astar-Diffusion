"""Command line entry point.

    guidance-field demo                      # self-contained demo, no data needed
    guidance-field experiment --maps 4       # generate maps, build fields, run all searches
    guidance-field field --map map.npy --trajectory traj.npy --out field.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _demo(args) -> int:
    from . import viz
    from .config import Config
    from .field import build_weighted_field
    from .experiment import build_case
    from .policies import (DistancePolicy, FieldOnlyPolicy, LinearFusionPolicy,
                           astar_search, priority_search)
    from .normalize import prepare_scalars
    from .sources import build_all_methods

    cfg = Config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    case = build_case(0, args.map_type, cfg.seed, cfg)
    grid = case["grid"]
    grade = args.grade if args.grade in case["trajectories"] else next(iter(case["trajectories"]))
    entry = case["trajectories"][grade]

    sources = build_all_methods(grid, entry["raw_path"], cfg)
    src = sources[cfg.source_method]
    field = build_weighted_field(grid, src, args.weight, cfg.lam)["field"]
    sc = prepare_scalars(grid, field, args.weight, case["goal"])

    print(f"map {case['scene_id']} ({case['map_type']}) {grid.height}x{grid.width}  "
          f"free={grid.free_fraction():.2f}  L*={case['L_star']:.0f}")
    print(f"trajectory: grade={grade}  collisions={entry['metrics']['collision_ratio']:.3f} "
          f"({entry['metrics']['collision_interval_count']} intervals)")
    print(f"sources: {len(src['positions_rc'])} kept ({cfg.source_method})  field: {args.weight}")

    rows = []
    for name, policy in (("A* (baseline)", None),
                         ("Distance-GBFS", DistancePolicy(sc)),
                         ("Field-Only", FieldOnlyPolicy(sc)),
                         ("Linear Fusion a=0.5", LinearFusionPolicy(sc, alpha=0.5))):
        if policy is None:
            res = astar_search(grid, case["start"], case["goal"])
        else:
            res = priority_search(grid, case["start"], case["goal"], policy)
        rows.append((name, res))
    print(f"\n{'method':22s} {'status':10s} {'expanded':>9s} {'path':>6s} {'ms':>7s}")
    from .policies import path_is_valid

    base = rows[0][1]["expanded"]
    for name, res in rows:
        length = res["g_cost"]
        if length is None:
            length = path_is_valid(grid, res["path"], case["start"], case["goal"]).get("length")
        print(f"{name:22s} {res['status']:10s} {res['expanded']:9d} "
              f"{(length or 0):6.0f} {res['search_time']*1000:7.2f}   ({res['expanded']/base:.2f}x A*)")

    viz.figure_field(out / "field.png", grid, field, args.weight, raw=entry["raw_path"],
                     intervals=entry["metrics"]["intervals"], sources=src["positions_rc"],
                     progress=src["progress"], start=case["start"], goal=case["goal"],
                     title=f"{case['scene_id']} · {grade} · {args.weight} guidance field",
                     caption=f"F(x) = max_i [ w(u_i) - lambda*d_free(x, s_i) ]   lambda={cfg.lam}")
    panels = []

    def panel(res, method):
        def draw(ax):
            viz.draw_occupancy(ax, case["occupancy"])
            if method != "A* (baseline)":
                viz._field_image(ax, field, args.weight)
            viz.draw_trajectory(ax, entry["raw_path"], entry["metrics"]["intervals"], lw=1.2)
            viz.draw_expansion(ax, grid, res.get("trace"))
            viz.draw_path(ax, grid, res["path"])
            viz.draw_endpoints(ax, case["start"], case["goal"])
            ax.set_title(f"{method}\nexpanded={res['expanded']}", fontsize=9)
        return draw

    for name, res in rows:
        panels.append(panel(res, name))
    viz.figure_grid(out / "search.png", panels, ncols=4,
                    suptitle=f"{case['scene_id']} · {grade} · expansion comparison")
    print(f"\nfigures -> {out / 'field.png'}, {out / 'search.png'}   ({time.perf_counter()-t0:.1f}s)")
    return 0


def _experiment(args) -> int:
    from .config import Config
    from .experiment import run_study

    cfg = Config(grid_size=args.size, seed=args.seed,
                 field_weights=tuple(args.weights.split(",")),
                 propagation=args.propagation)
    summary = run_study(cfg, outdir=args.out, n_instances=args.maps,
                        map_types=tuple(t for t in args.types.replace(",", "")), make_figures=args.figures)
    t = summary["totals"]
    print(f"\nmaps={t['maps']} runs={t['runs']} success={t['success']} valid={t['valid_paths']} "
          f"({summary['runtime_s']}s)")
    agg = summary["by_method_field"]
    print(f"\n{'method|field':34s} {'expanded':>9s} {'search_ms':>10s} {'total_ms':>9s} {'L/L*':>6s}")
    for key in sorted(agg, key=lambda k: agg[k]["total_time"]["mean"]):
        b = agg[key]
        print(f"{key:34s} {b['expanded']['mean']:9.1f} {b['search_time']['mean']*1000:10.3f} "
              f"{b['total_time']['mean']*1000:9.3f} {b.get('optimality_ratio', {}).get('mean', float('nan')):6.3f}")
    print(f"\n-> {Path(args.out) / 'analysis' / 'search_results.csv'}")
    print(f"-> {Path(args.out) / 'summary.json'}")
    return 0


def _field(args) -> int:
    from . import viz
    from .config import Config
    from .field import build_weighted_field
    from .grid import Grid

    cfg = Config()
    occupancy = np.load(args.map)
    raw = np.load(args.trajectory) if args.trajectory else None
    grid = Grid(occupancy)
    start = tuple(args.start) if args.start else tuple(int(v) for v in np.argwhere(grid.free)[0])
    goal = tuple(args.goal) if args.goal else tuple(int(v) for v in np.argwhere(grid.free)[-1])
    from .sources import build_all_methods

    if raw is None:
        from .paths import astar
        nodes = astar(grid, grid.idx(*start), grid.idx(*goal))
        raw = np.array([grid.rc(int(i)) for i in nodes], dtype=float)
    sources = build_all_methods(grid, raw, cfg)
    src = sources[args.method]
    field = build_weighted_field(grid, src, args.weight, cfg.lam)["field"]
    intervals = []
    from .collision import detect_collision_intervals
    intervals = detect_collision_intervals(grid, raw)["intervals"]
    viz.figure_field(args.out, grid, field, args.weight, raw=raw, intervals=intervals,
                     sources=src["positions_rc"], progress=src["progress"], start=start, goal=goal,
                     title=f"guidance field [{args.weight}] from {args.map}",
                     caption=f"sources={len(src['positions_rc'])} ({args.method})  lambda={cfg.lam}")
    print(f"-> {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="guidance-field", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="self-contained demo (generates its own map)")
    d.add_argument("--out", default="outputs/demo")
    d.add_argument("--map-type", default="C", choices=list("ABCD"))
    d.add_argument("--grade", default="moderate_collision")
    d.add_argument("--weight", default="linear", choices=("uniform", "linear", "quadratic"))
    d.set_defaults(func=_demo)

    e = sub.add_parser("experiment", help="run the full scenario study")
    e.add_argument("--out", default="outputs/study")
    e.add_argument("--maps", type=int, default=2, help="instances per map family")
    e.add_argument("--types", default="A,B,C,D")
    e.add_argument("--size", type=int, default=64)
    e.add_argument("--seed", type=int, default=20260214)
    e.add_argument("--weights", default="linear,quadratic")
    e.add_argument("--propagation", default="auto",
                   choices=("auto", "scipy", "vectorized", "heap", "bucket"))
    e.add_argument("--figures", action="store_true")
    e.set_defaults(func=_experiment)

    f = sub.add_parser("field", help="build a field from a .npy map (+ optional trajectory)")
    f.add_argument("--map", required=True)
    f.add_argument("--trajectory", default=None)
    f.add_argument("--out", default="field.png")
    f.add_argument("--weight", default="linear", choices=("uniform", "linear", "quadratic"))
    f.add_argument("--method", default="M0", choices=("M0", "M1", "M2"))
    f.add_argument("--start", type=int, nargs=2, default=None)
    f.add_argument("--goal", type=int, nargs=2, default=None)
    f.set_defaults(func=_field)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
