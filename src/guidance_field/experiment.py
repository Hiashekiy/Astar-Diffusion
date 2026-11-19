"""End-to-end pipeline: map -> graded colliding trajectories -> sources -> fields -> search.

``build_case`` gives you everything for one map, ``run_case`` runs every search configuration
on it, and ``run_study`` loops over a whole scenario set and writes a CSV + summary JSON.
Nothing here needs the historical experiment outputs: a case is generated from a seed.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .config import MAP_TYPES, METHODS, SOURCE_METHODS, TRAJECTORY_GRADES, WEIGHT_KINDS, Config
from .field import source_weights
from .grid import Grid, detour_profile
from .metrics import channel_metrics, compare_to_baseline, field_metrics, source_metrics, summarize
from .normalize import prepare_scalars
from .policies import (
    AdaptiveFusionPolicy,
    DistancePolicy,
    FieldOnlyPolicy,
    LinearFusionPolicy,
    NonlinearFusionPolicy,
    PriorityPolicy,
    astar_search,
    dijkstra_length,
    path_is_valid,
    priority_search,
)
from .propagation import propagate as propagate_field
from .scenes import generate_scene
from .sources import build_all_methods
from .trajectories import generate_trajectories

CSV_COLUMNS = [
    "scene_id", "map_type", "grade", "field", "method", "alpha", "gamma", "status",
    "expanded", "generated", "search_time", "field_build_time", "total_time",
    "path_length", "L_star", "optimality_ratio", "path_valid", "success",
    "collision_ratio", "collision_interval_count", "num_sources",
    "mean_field_on_expanded", "mean_distance_on_expanded",
]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------------- #
# one map
# --------------------------------------------------------------------------- #
def build_case(index: int, map_type: str, seed: int | None = None, cfg: Config | None = None) -> dict:
    """Generate one map with its graded trajectories (feasible .. severe collision)."""
    cfg = cfg or Config()
    seed = cfg.seed + 1000 * MAP_TYPES.index(map_type) + index if seed is None else seed
    scene = generate_scene(index, map_type, seed, cfg)
    grid = Grid(scene["occupancy"])
    meta = scene["metadata"]
    start = tuple(meta["start"])
    goal = tuple(meta["goal"])
    profile = detour_profile(grid, start, goal)
    detour = np.where(np.isfinite(profile["detour"]), profile["detour"], np.inf)
    rng = np.random.default_rng(seed + 7919)
    produced = generate_trajectories(grid, scene["occupancy"], start, goal, cfg, rng)
    return {
        "scene_id": meta["scene_id"],
        "map_type": map_type,
        "meta": meta,
        "grid": grid,
        "occupancy": scene["occupancy"],
        "start": start,
        "goal": goal,
        "detour": detour,
        "L_star": dijkstra_length(grid, start, goal),
        "trajectories": produced["trajectories"],
        "reference": produced["reference"],
        "coverage": produced["coverage"],
    }


# --------------------------------------------------------------------------- #
# every configuration on one map
# --------------------------------------------------------------------------- #
def _diagnostics(grid: Grid, trace, scalars: dict) -> tuple[float | None, float | None]:
    if not trace:
        return None, None
    rows, cols = np.divmod(np.asarray(trace, dtype=int), grid.width)
    f_vals = scalars["F_hat"][rows, cols]
    f_vals = f_vals[np.isfinite(f_vals)]
    d_vals = scalars["d_hat"][rows, cols]
    return (float(f_vals.mean()) if f_vals.size else None, float(d_vals.mean()))


def run_case(case: dict, cfg: Config | None = None, record_traces: bool = False) -> tuple[list[dict], dict]:
    """Run B0-B5 (x weights x alphas) on one case; returns (records, traces)."""
    cfg = cfg or Config()
    grid = case["grid"]
    start, goal = case["start"], case["goal"]
    L_star = case["L_star"]
    masks = {
        "start_ball": grid.geodesic_ball(start, cfg.ball_radius),
        "goal_ball": grid.geodesic_ball(goal, cfg.ball_radius),
    }
    grades = [g for g in TRAJECTORY_GRADES if g in case["trajectories"]]
    records: list[dict] = []
    traces: dict = {}

    for grade in grades:
        entry = case["trajectories"][grade]
        raw = entry["raw_path"]
        intervals = entry["metrics"]["intervals"]
        sources = build_all_methods(grid, raw, cfg)
        entry["num_sources"] = int(len(sources[cfg.source_method]["positions_rc"]))

        scalars_by_weight, build_time = {}, {}
        for weight in cfg.field_weights:
            src = sources[cfg.source_method]
            weights = source_weights(src["progress"], weight)
            t0 = time.perf_counter()
            field = propagate_field(grid, src["positions_rc"], weights, cfg.lam, method=cfg.propagation)
            build_time[weight] = time.perf_counter() - t0
            scalars_by_weight[weight] = prepare_scalars(grid, field, weight, goal)
            scalars_by_weight[weight]["field"] = field

        # B0 / B1 do not use the field: run once per (map, grade)
        b0 = astar_search(grid, start, goal, cutoff=cfg.node_cutoff)
        v0 = path_is_valid(grid, b0["path"], start, goal)
        records.append(_record(case, grade, "none", "B0_astar", None, None, b0, v0, L_star, 0.0, (None, None)))
        if record_traces:
            traces[(grade, "none", "B0_astar", None)] = b0

        any_sc = scalars_by_weight[cfg.field_weights[0]]
        b1 = priority_search(grid, start, goal, DistancePolicy(any_sc), cfg.node_cutoff)
        v1 = path_is_valid(grid, b1["path"], start, goal)
        records.append(_record(case, grade, "none", "B1_distance_gbfs", None, None, b1, v1, L_star, 0.0,
                               _diagnostics(grid, b1["trace"], any_sc)))
        if record_traces:
            traces[(grade, "none", "B1_distance_gbfs", None)] = b1

        for weight, sc in scalars_by_weight.items():
            runs: list[tuple[str, float | None, float | None, PriorityPolicy]] = []
            from .policies import FieldOnlyPolicy

            runs.append(("B2_field_only", None, None, FieldOnlyPolicy(sc)))
            for alpha in cfg.alphas:
                runs.append(("B3_linear_fusion", alpha, 1.0, LinearFusionPolicy(sc, alpha=alpha)))
            for alpha in cfg.alphas:
                runs.append(("B4_nonlinear_fusion", alpha, cfg.gamma_main,
                             NonlinearFusionPolicy(sc, alpha=alpha, gamma=cfg.gamma_main)))
            runs.append(("B5_adaptive_fusion", None, cfg.adaptive_gamma,
                         AdaptiveFusionPolicy(sc, alpha_far=cfg.alpha_far, alpha_near=cfg.alpha_near,
                                              eta=cfg.adaptive_eta, gamma=cfg.adaptive_gamma)))
            for method, alpha, gamma, policy in runs:
                res = priority_search(grid, start, goal, policy, cfg.node_cutoff)
                valid = path_is_valid(grid, res["path"], start, goal)
                diag = _diagnostics(grid, res["trace"], sc)
                rec = _record(case, grade, weight, method, alpha, gamma, res, valid, L_star,
                              build_time[weight], diag)
                fm = field_metrics(grid, sc["field"], np.full(grid.free.shape, -1, dtype=np.int32),
                                   sources[cfg.source_method], masks, cfg.lam)
                cm = channel_metrics(grid, sc["field"], case["detour"], cfg)
                rec.update({f"field_{k}": v for k, v in fm.items() if k in ("owner_entropy", "delta_F")})
                rec["channel_auc"] = cm["channel_auc"]
                records.append(rec)
                if record_traces:
                    traces[(grade, weight, method, alpha)] = res
    return records, traces


def _record(case, grade, field_weight, method, alpha, gamma, res, valid, L_star, build_time, diag) -> dict:
    length = valid.get("length") if valid.get("valid") else None
    tm = case["trajectories"][grade]["metrics"]
    return {
        "scene_id": case["scene_id"],
        "map_type": case["map_type"],
        "grade": grade,
        "field": field_weight,
        "method": method,
        "alpha": alpha,
        "gamma": gamma,
        "status": res["status"],
        "expanded": res["expanded"],
        "generated": res["generated"],
        "search_time": res["search_time"],
        "field_build_time": build_time,
        "total_time": build_time + res["search_time"],
        "path_length": length,
        "L_star": L_star,
        "optimality_ratio": (length / L_star) if (length is not None and L_star) else None,
        "path_valid": bool(valid.get("valid")),
        "success": res["status"] == "success",
        "collision_ratio": tm["collision_ratio"],
        "collision_interval_count": tm["collision_interval_count"],
        "num_sources": int(case["trajectories"][grade].get("num_sources", 0)),
        "mean_field_on_expanded": diag[0],
        "mean_distance_on_expanded": diag[1],
    }


# --------------------------------------------------------------------------- #
# whole study
# --------------------------------------------------------------------------- #
def run_study(cfg: Config | None = None, outdir: str | Path = "outputs/study",
              n_instances: int | None = None, map_types: tuple[str, ...] | None = None,
              make_figures: bool = False, verbose: bool = True) -> dict:
    """Loop over map families, run every configuration, write CSV + summary JSON."""
    cfg = cfg or Config()
    n = cfg.instances_per_type if n_instances is None else n_instances
    map_types = tuple(map_types or MAP_TYPES)
    root = Path(outdir)
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    if make_figures:
        (root / "figures").mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    records: list[dict] = []
    cases: list[dict] = []
    index = 0
    for map_type in map_types:
        for k in range(n):
            case = build_case(k, map_type, cfg.seed + 1000 * MAP_TYPES.index(map_type) + k, cfg)
            recs, traces = run_case(case, cfg, record_traces=make_figures)
            records.extend(recs)
            cases.append(case)
            index += 1
            if verbose:
                print(f"[{case['scene_id']}] {map_type}  L*={case['L_star']}  "
                      f"grades={len(case['trajectories'])}  runs={len(recs)}", flush=True)
            if make_figures:
                _case_figures(root / "figures", case, cfg, traces)

    with (root / "analysis" / "search_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in CSV_COLUMNS})

    summary = {
        "config": cfg.as_dict(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round(time.perf_counter() - t_start, 2),
        "totals": {
            "maps": len(cases),
            "runs": len(records),
            "success": sum(1 for r in records if r["success"]),
            "valid_paths": sum(1 for r in records if r["path_valid"]),
        },
        "by_method_field": summarize(records, ("method", "field")),
        "by_grade": summarize(records, ("method", "field", "grade")),
        "by_alpha": summarize([r for r in records if r["method"] in ("B3_linear_fusion", "B4_nonlinear_fusion")],
                              ("method", "field", "alpha")),
        "vs_baseline": compare_to_baseline(records),
    }
    (root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return summary


def _case_figures(outdir: Path, case: dict, cfg: Config, traces: dict) -> None:
    from . import viz

    grade = cfg.core_grade if cfg.core_grade in case["trajectories"] else next(iter(case["trajectories"]))
    entry = case["trajectories"][grade]
    weight = cfg.field_weights[0]
    src = build_all_methods(case["grid"], entry["raw_path"], cfg)[cfg.source_method]
    field = propagate_field(case["grid"], src["positions_rc"], source_weights(src["progress"], weight),
                            cfg.lam, method=cfg.propagation)
    viz.figure_field(outdir / f"{case['scene_id']}_field.png", case["grid"], field, weight,
                     raw=entry["raw_path"], intervals=entry["metrics"]["intervals"],
                     sources=src["positions_rc"], progress=src["progress"],
                     start=case["start"], goal=case["goal"],
                     title=f"{case['scene_id']} · {case['map_type']} · {grade} · {weight} field",
                     caption=f"F(x) = max_i [ w(u_i) - lambda*d_free(x, s_i) ]   lambda={cfg.lam}")
    panels = []
    for method, alpha in (("B0_astar", None), ("B1_distance_gbfs", None), ("B2_field_only", None),
                          ("B3_linear_fusion", 0.5), ("B4_nonlinear_fusion", 0.5), ("B5_adaptive_fusion", None)):
        res = traces.get((grade, weight, method, alpha)) or traces.get((grade, "none", method, None))
        if res is None:
            continue

        def draw(ax, res=res, method=method, alpha=alpha):
            viz.draw_occupancy(ax, case["occupancy"])
            if method not in ("B0_astar", "B1_distance_gbfs"):
                viz._field_image(ax, field, weight)
            viz.draw_trajectory(ax, entry["raw_path"], entry["metrics"]["intervals"], lw=1.2)
            viz.draw_expansion(ax, case["grid"], res.get("trace"))
            viz.draw_path(ax, case["grid"], res["path"])
            viz.draw_endpoints(ax, case["start"], case["goal"])
            ax.set_title(f"{method}" + (f" alpha={alpha}" if alpha else "") + f"\nexpanded={res['expanded']}",
                         fontsize=8.5)

        panels.append(draw)
    if panels:
        viz.figure_grid(outdir / f"{case['scene_id']}_search.png", panels, ncols=3,
                        suptitle=f"{case['scene_id']} · {case['map_type']} · {grade}   expansion comparison")
