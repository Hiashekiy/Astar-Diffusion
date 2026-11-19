"""Search-stage pipeline: Stage-2A artefacts -> guidance fields -> B0-B5 -> metrics + figures.

Runs on the maps, trajectories and M0 sources already produced by stage 2A.  Fields are
rebuilt in-process from the M0 sources so that "Field Build Time" is a real measurement
rather than the cost of loading a .npy file; the rebuilt field is then checked against the
stored one so the two stages cannot drift apart.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from guidance_stage1.grid import Grid
from guidance_stage2.config import Stage2Config
from guidance_stage2.config import Stage2Config as _S2Config
from guidance_stage2.fields import source_weights
from guidance_stage2.propagation import propagate_auto
from guidance_stage3 import visualization as viz
from guidance_stage3.config import METHODS, SearchConfig
from guidance_stage3.metrics import compare_to_baseline, summarize
from guidance_stage3.normalize import prepare_scalars
from guidance_stage3.search import (
    POLICY_CLASSES,
    AdaptiveFusionPolicy,
    LinearFusionPolicy,
    NonlinearFusionPolicy,
    astar_search,
    dijkstra_length,
    path_is_valid,
    priority_search,
)

CSV_COLUMNS = [
    "scene_id", "map_type", "grade", "field", "method", "alpha", "gamma",
    "status", "expanded", "generated", "search_time", "field_build_time", "total_time",
    "path_length", "L_star", "optimality_ratio", "path_valid", "success",
    "mean_field_on_expanded", "mean_distance_on_expanded", "num_sources",
    "collision_ratio",
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


def _load_scene(root: Path, sid: str) -> dict:
    occ = np.load(root / "scenes" / sid / "occupancy.npy")
    sg = np.load(root / "scenes" / sid / "start_goal.npy")
    md = json.loads((root / "scenes" / sid / "metadata.json").read_text(encoding="utf-8"))
    tm = json.loads((root / "trajectories" / sid / "metadata.json").read_text(encoding="utf-8"))
    return {
        "scene_id": sid,
        "occupancy": occ,
        "grid": Grid(occ),
        "start": (int(sg[0][0]), int(sg[0][1])),
        "goal": (int(sg[1][0]), int(sg[1][1])),
        "meta": md,
        "traj_meta": tm,
    }


def _diagnostics(grid: Grid, trace: list[int], scalars: dict, goal) -> tuple[float | None, float | None]:
    if not trace:
        return None, None
    rows, cols = np.divmod(np.asarray(trace, dtype=int), grid.width)
    f_vals = scalars["F_hat"][rows, cols]
    d_vals = scalars["d_hat"][rows, cols]
    f_vals = f_vals[np.isfinite(f_vals)]
    return (float(f_vals.mean()) if f_vals.size else None, float(d_vals.mean()))


def _record(scene, grade, field_weight, method, alpha, gamma, run, valid, L_star, build_time, diag) -> dict:
    path = run["path"]
    length = valid.get("length") if valid.get("valid") else None
    return {
        "scene_id": scene["scene_id"],
        "map_type": scene["meta"]["type"],
        "grade": grade,
        "field": field_weight if field_weight else "none",
        "method": method,
        "alpha": alpha,
        "gamma": gamma,
        "status": run["status"],
        "expanded": run["expanded"],
        "generated": run["generated"],
        "search_time": run["search_time"],
        "field_build_time": build_time or 0.0,
        "total_time": (build_time or 0.0) + run["search_time"],
        "path_length": length,
        "L_star": L_star,
        "optimality_ratio": (length / L_star) if (length is not None and L_star) else None,
        "path_valid": bool(valid.get("valid")),
        "success": run["status"] == "success",
        "mean_field_on_expanded": diag[0],
        "mean_distance_on_expanded": diag[1],
        "num_sources": scene.get("num_sources"),
        "collision_ratio": scene["traj_meta"]["trajectories"][grade]["metrics"]["collision_ratio"],
        "_trace": run.get("trace"),
    }


def run(cfg: SearchConfig, make_figures: bool = True, scene_limit: int | None = None) -> dict:
    t_start = time.perf_counter()
    s2 = Path(cfg.stage2_outdir)
    if not s2.is_absolute():
        s2 = Path(__file__).resolve().parents[2] / s2
    root = Path(cfg.outdir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    for name in ("figures", "figures/summary", "analysis"):
        (root / name).mkdir(parents=True, exist_ok=True)

    scene_ids = sorted(p.name for p in (s2 / "scenes").iterdir() if p.is_dir())
    if scene_limit:
        scene_ids = scene_ids[:scene_limit]

    records: list[dict] = []
    traces: dict[tuple, dict] = {}
    lam = Stage2Config.lam
    field_max_diff = 0.0

    for sid in scene_ids:
        scene = _load_scene(s2, sid)
        grid = scene["grid"]
        start, goal = scene["start"], scene["goal"]
        L_star = dijkstra_length(grid, start, goal)
        scene["L_star"] = L_star
        grades = [g for g in cfg.figure_grades if g in scene["traj_meta"]["trajectories"]]

        for grade in grades:
            t_build_total = {}
            scalars_by_weight = {}
            for weight in cfg.field_weights:
                z = np.load(s2 / "sources" / sid / grade / f"{cfg.source_method}_sources.npz")
                sources = {"positions_rc": z["positions_rc"], "progress": z["progress"]}
                scene["num_sources"] = int(len(z["positions_rc"]))
                t0 = time.perf_counter()
                field_arr = propagate_auto(grid, sources["positions_rc"],
                                           source_weights(sources["progress"], weight), lam)
                t_build_total[weight] = time.perf_counter() - t0
                built = {"field": field_arr}
                stored = np.load(s2 / "fields" / sid / grade / f"{cfg.source_method}_{weight}.npy").astype(np.float64)
                ok = grid.free & np.isfinite(stored) & np.isfinite(built["field"])
                if ok.any():
                    field_max_diff = max(field_max_diff, float(np.max(np.abs(stored[ok] - built["field"][ok]))))
                scalars_by_weight[weight] = prepare_scalars(grid, built["field"], weight, goal)

            # B0 / B1 are field-independent: run once per (scene, grade)
            b0 = astar_search(grid, start, goal, cutoff=cfg.node_cutoff)
            v0 = path_is_valid(grid, b0["path"], start, goal)
            records.append(_record(scene, grade, None, "B0_astar", None, None, b0, v0, L_star, None, (None, None)))
            if grade == cfg.core_grade:
                traces[(sid, "none", "B0_astar", None)] = b0
                traces[(sid, "none", "B1_distance_gbfs", None)] = b1_pending = None

            sc_any = scalars_by_weight[cfg.field_weights[0]]
            b1 = priority_search(grid, start, goal, POLICY_CLASSES["B1_distance_gbfs"](sc_any), cfg.node_cutoff)
            v1 = path_is_valid(grid, b1["path"], start, goal)
            records.append(_record(scene, grade, None, "B1_distance_gbfs", None, None, b1, v1, L_star, None,
                                   _diagnostics(grid, b1["trace"], sc_any, goal)))
            if grade == cfg.core_grade:
                traces[(sid, "none", "B1_distance_gbfs", None)] = b1

            for weight, sc in scalars_by_weight.items():
                runs = []
                runs.append(("B2_field_only", None, None, POLICY_CLASSES["B2_field_only"](sc)))
                for alpha in cfg.alphas:
                    runs.append(("B3_linear_fusion", alpha, 1.0, LinearFusionPolicy(sc, alpha=alpha)))
                for alpha in cfg.alphas:
                    runs.append(("B4_nonlinear_fusion", alpha, cfg.gamma_main,
                                 NonlinearFusionPolicy(sc, alpha=alpha, gamma=cfg.gamma_main)))
                runs.append(("B5_adaptive_fusion", None, cfg.adaptive_gamma,
                             AdaptiveFusionPolicy(sc, alpha_far=cfg.alpha_far, alpha_near=cfg.alpha_near,
                                                  eta=cfg.adaptive_eta, gamma=cfg.adaptive_gamma)))
                for method, alpha, gamma, policy in runs:
                    run_res = priority_search(grid, start, goal, policy, cfg.node_cutoff)
                    valid = path_is_valid(grid, run_res["path"], start, goal)
                    diag = _diagnostics(grid, run_res["trace"], sc, goal)
                    records.append(_record(scene, grade, weight, method, alpha, gamma, run_res, valid,
                                           L_star, t_build_total[weight], diag))
                    if grade == cfg.core_grade:
                        traces[(sid, weight, method, alpha)] = run_res

        print(f"[{sid}] {scene['meta']['type']}  L*={L_star}  runs={sum(1 for r in records if r['scene_id'] == sid)}",
              flush=True)

    # ------------------------------------------------------------------ figures
    if make_figures:
        fig_dir = root / "figures"
        for sid in scene_ids:
            scene = _load_scene(s2, sid)
            ctx = _figure_context(scene, cfg.core_grade, s2, cfg, traces)
            if ctx is None:
                continue
            viz.figure_search_overlay(fig_dir / f"search_overlay_{sid}.png", ctx)
            viz.figure_expansion_compare(fig_dir / f"expansion_compare_{sid}.png", ctx,
                                         cfg.field_weights[0], 0.5)
        print(f"[figures] {len(scene_ids)} scenes rendered", flush=True)

    # ------------------------------------------------------------------ outputs
    with (root / "analysis" / "search_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in CSV_COLUMNS})

    by_method = summarize(records, ("method", "field"))
    by_grade = summarize(records, ("method", "field", "grade"))
    by_alpha = summarize([r for r in records if r["method"] in ("B3_linear_fusion", "B4_nonlinear_fusion")],
                         ("method", "field", "alpha"))
    baseline = compare_to_baseline(records)

    if make_figures and records:
        viz.summary_figures(root / "figures" / "summary", records, by_grade, by_method)

    summary = {
        "config": cfg.as_dict(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round(time.perf_counter() - t_start, 2),
        "totals": {
            "maps": len(scene_ids),
            "runs": len(records),
            "success": sum(1 for r in records if r["success"]),
            "status_counts": {
                s: sum(1 for r in records if r["status"] == s) for s in ("success", "unreachable", "cutoff")
            },
            "paths_valid": sum(1 for r in records if r["path_valid"]),
        },
        "field_rebuild_max_abs_diff_vs_stage2": field_max_diff,
        "aggregate": {"by_method_field": by_method, "by_grade": by_grade, "by_alpha": by_alpha},
        "vs_baseline": baseline,
    }
    (root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return summary

def _figure_context(scene: dict, grade: str, s2: Path, cfg: SearchConfig, traces: dict) -> dict | None:
    """Everything the per-scene figures need, rebuilt from the stored artefacts."""
    grid, start, goal = scene["grid"], scene["start"], scene["goal"]
    weight0 = cfg.field_weights[0]
    if (scene["scene_id"], weight0, "B3_linear_fusion", 0.5) not in traces:
        return None
    z = np.load(s2 / "sources" / scene["scene_id"] / grade / f"{cfg.source_method}_sources.npz")
    fields = {
        w: np.load(s2 / "fields" / scene["scene_id"] / grade / f"{cfg.source_method}_{w}.npy").astype(np.float64)
        for w in cfg.field_weights
    }
    results = {}
    for (sid, weight, method, alpha), res in traces.items():
        if sid != scene["scene_id"]:
            continue
        length = path_is_valid(grid, res["path"], start, goal).get("length")
        results[(weight, method, alpha)] = {
            "expanded": res["expanded"], "generated": res["generated"],
            "length": length, "trace": res.get("trace", []), "path": res["path"],
        }
    raw = np.load(s2 / "trajectories" / scene["scene_id"] / f"{grade}.npy")
    return {
        "scene_id": scene["scene_id"], "map_type": scene["meta"]["type"], "grade": grade,
        "grid": grid, "occupancy": scene["occupancy"], "start": start, "goal": goal,
        "raw": raw, "sources": z["positions_rc"], "progress": z["progress"],
        "fields": fields, "results": results, "weights": cfg.field_weights,
        "L_star": scene.get("L_star") or dijkstra_length(grid, start, goal),
    }

