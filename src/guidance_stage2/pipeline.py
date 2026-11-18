"""Stage-2A pipeline: maps -> graded colliding trajectories -> M0/M1/M2 -> fields -> figures."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from guidance_stage1.grid import Grid, detour_profile
from guidance_stage2 import visualization as viz
from guidance_stage2.config import (
    MAP_TYPES,
    SOURCE_METHODS,
    TRAJECTORY_GRADES,
    WEIGHT_KINDS,
    Stage2Config,
)
from guidance_stage2.fields import build_all_weight_variants
from guidance_stage2.metrics import (
    channel_metrics,
    compare_to_reference,
    corridor_mask,
    field_metrics,
    source_metrics,
)
from guidance_stage2.scenes import generate_scene
from guidance_stage2.source_methods import build_all_methods
from guidance_stage2.trajectories import generate_trajectories

CSV_COLUMNS = [
    "scene_id", "map_type", "grade", "generator", "collision_ratio", "collision_interval_count",
    "max_collision_interval_length", "path_length", "rho", "method", "weight",
    "num_valid_sources", "valid_source_ratio", "num_duplicate_source_cells", "num_invalid_sources",
    "mean_projection_distance", "max_projection_distance", "num_collision_intervals",
    "num_bridged_intervals", "bridge_success_rate", "bridge_total_length", "bridge_detour_ratio",
    "roi_expansion_count", "progress_covered_fraction",
    "field_min", "field_max", "field_mean", "field_std", "owner_distinct", "owner_entropy",
    "owner_dominant_fraction", "F_start_ball", "F_goal_ball", "delta_F", "monotone_fraction",
    "profile_slope", "lipschitz_violation", "free_cell_finite_ratio", "obstacle_nan_ratio",
    "F_near_channel", "F_off_channel", "channel_gap", "channel_auc",
    "mae_all_free", "corr_all_free", "mae_near_reference_path", "channel_mean_diff",
    "high_response_coverage",
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


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _aggregate(records: list[dict]) -> dict:
    keys = (
        "delta_F", "owner_distinct", "owner_entropy", "owner_dominant_fraction", "monotone_fraction",
        "profile_slope", "field_min", "field_max", "field_mean", "field_std", "lipschitz_violation",
        "free_cell_finite_ratio", "obstacle_nan_ratio", "F_near_channel", "F_off_channel",
        "channel_gap", "channel_auc", "mae_all_free", "corr_all_free", "mae_near_reference_path",
        "high_response_coverage", "num_valid_sources", "valid_source_ratio",
        "progress_covered_fraction", "bridge_success_rate", "num_duplicate_source_cells",
        "num_invalid_sources", "mean_projection_distance",
    )
    out: dict[str, dict] = {}
    for grade in TRAJECTORY_GRADES:
        for method in SOURCE_METHODS:
            for weight in WEIGHT_KINDS:
                subset = [
                    r for r in records
                    if r["grade"] == grade and r["method"] == method and r["weight"] == weight
                ]
                if not subset:
                    continue
                bucket = {"n": len(subset)}
                for key in keys:
                    vals = np.array([float(r[key]) for r in subset if r.get(key) is not None])
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        bucket[key] = {
                            "mean": float(vals.mean()),
                            "std": float(vals.std()),
                            "min": float(vals.min()),
                            "max": float(vals.max()),
                        }
                out[f"{grade}|{method}|{weight}"] = bucket
    return out


def run(cfg: Stage2Config, make_figures: bool = True) -> dict:
    t_start = time.time()
    root = Path(cfg.outdir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    for name in ("scenes", "trajectories", "sources", "fields", "figures", "figures/summary", "analysis"):
        (root / name).mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    scene_summaries: list[dict] = []
    gallery: list[dict] = []

    for t_i, map_type in enumerate(MAP_TYPES):
        for k in range(cfg.instances_per_type):
            scene_index = t_i * cfg.instances_per_type + k
            seed = cfg.seed + 1000 * t_i + k
            scene = generate_scene(scene_index, map_type, seed, cfg)
            md = scene["metadata"]
            occ = scene["occupancy"]
            grid = Grid(occ)
            start = tuple(md["start"])
            goal = tuple(md["goal"])
            sid = md["scene_id"]

            for sub in ("scenes", "trajectories", "sources", "fields", "figures"):
                (root / sub / sid).mkdir(parents=True, exist_ok=True)

            np.save(root / "scenes" / sid / "occupancy.npy", occ)
            np.save(root / "scenes" / sid / "start_goal.npy", np.array([start, goal], dtype=np.int64))
            _write_json(root / "scenes" / sid / "metadata.json", md)

            profile = detour_profile(grid, start, goal)
            detour = profile["detour"]
            detour = np.where(np.isfinite(detour), detour, np.inf)
            masks = {
                "start_ball": grid.geodesic_ball(start, cfg.ball_radius),
                "goal_ball": grid.geodesic_ball(goal, cfg.ball_radius),
            }

            rng = np.random.default_rng(seed + 7919)
            tr = generate_trajectories(grid, occ, start, goal, cfg, rng)
            np.save(root / "trajectories" / sid / "reference_astar.npy", tr["reference"]["polyline"])
            traj_meta = {
                "scene_id": sid,
                "map_type": map_type,
                "seed": seed,
                "reference": {
                    "length": tr["reference"]["length"],
                    "num_nodes": tr["reference"]["num_nodes"],
                    "generator": "astar",
                },
                "coverage": tr["coverage"],
                "trajectories": {},
            }
            for grade, entry in tr["trajectories"].items():
                stored = np.asarray(entry["raw_path"], dtype=np.float64)
                np.save(root / "trajectories" / sid / f"{grade}.npy", stored)
                traj_meta["trajectories"][grade] = {
                    "generator": entry["generator"],
                    "params": entry["params"],
                    "num_stored_points": int(len(stored)),
                    "metrics": {k: v for k, v in entry["metrics"].items() if k != "intervals"},
                    "collision_intervals": entry["metrics"]["intervals"],
                }
            _write_json(root / "trajectories" / sid / "metadata.json", traj_meta)

            # ---- sources + fields for every (grade, method, weight) ----
            cases: dict[tuple, dict] = {}
            source_store: dict[tuple, dict] = {}
            for grade, entry in tr["trajectories"].items():
                sources = build_all_methods(grid, entry["raw_path"], cfg)
                src_dir = root / "sources" / sid / grade
                src_dir.mkdir(parents=True, exist_ok=True)
                bridge_meta = {}
                for method in SOURCE_METHODS:
                    src = sources[method]
                    source_store[(grade, method)] = src
                    weights = {
                        kind: np.asarray(_weights_for(src, kind), dtype=np.float64) for kind in WEIGHT_KINDS
                    }
                    support = src.get("support_path")
                    np.savez(
                        src_dir / f"{method}_sources.npz",
                        positions_rc=np.asarray(src["positions_rc"], dtype=np.int64),
                        progress=np.asarray(src["progress"], dtype=np.float64),
                        source_origin=np.asarray(src["origin"], dtype="U16"),
                        interval_id=np.asarray(src["interval_id"], dtype=np.int64),
                        weights_uniform=weights["uniform"],
                        weights_linear=weights["linear"],
                        weights_quadratic=weights["quadratic"],
                        support_path=np.asarray(support if support is not None else np.zeros((0, 2)), dtype=np.float64),
                    )
                    bridge_meta[method] = {
                        "metadata": src["metadata"],
                        "support_length": src["metadata"].get("support_length"),
                    }
                    variants = build_all_weight_variants(grid, src, cfg.lam)
                    field_dir = root / "fields" / sid / grade
                    field_dir.mkdir(parents=True, exist_ok=True)
                    for kind in WEIGHT_KINDS:
                        np.save(field_dir / f"{method}_{kind}.npy", variants[kind]["field"].astype(np.float32))
                        cases[(grade, method, kind)] = {
                            "field": variants[kind]["field"],
                            "owner": variants[kind]["owner"],
                            "source_progress": np.asarray(src["progress"], dtype=np.float64),
                            "support_path": src.get("support_path"),
                            "sources": np.asarray(src["positions_rc"], dtype=int),
                            "num_sources": variants[kind]["num_sources"],
                        }
                _write_json(src_dir / "bridge_metadata.json", bridge_meta)

            # ---- reference field (collision-free feasible trajectory, same weights) ----
            ref_grade = "feasible" if "feasible" in tr["trajectories"] else next(iter(tr["trajectories"]))
            ref_field = {kind: cases[(ref_grade, "M0", kind)]["field"] for kind in WEIGHT_KINDS}
            ref_mask = corridor_mask(grid, tr["reference"]["nodes"])

            # ---- metrics ----
            scene_records = 0
            for grade, entry in tr["trajectories"].items():
                tm = entry["metrics"]
                for method in SOURCE_METHODS:
                    src = source_store[(grade, method)]
                    sm = source_metrics(src)
                    for kind in WEIGHT_KINDS:
                        case = cases[(grade, method, kind)]
                        fm = field_metrics(grid, case["field"], case["owner"], src, masks, cfg.lam)
                        cm = channel_metrics(grid, case["field"], detour, cfg)
                        rm = compare_to_reference(grid, case["field"], ref_field[kind], ref_mask)
                        case["metrics"] = {**fm, **cm, **rm}
                        record = {
                            "scene_id": sid,
                            "map_type": map_type,
                            "grade": grade,
                            "generator": entry["generator"],
                            "method": method,
                            "weight": kind,
                            "collision_ratio": tm["collision_ratio"],
                            "collision_interval_count": tm["collision_interval_count"],
                            "max_collision_interval_length": tm["max_collision_interval_length"],
                            "path_length": tm["path_length"],
                            "rho": tm["rho"],
                            **sm,
                            **fm,
                            **cm,
                            **rm,
                        }
                        records.append(record)
                        scene_records += 1

            # ---- figures ----
            if make_figures:
                fdir = root / "figures" / sid
                grade_order = [g for g in TRAJECTORY_GRADES if g in tr["trajectories"]]
                viz.figure_trajectories(fdir / "trajectories.png", occ, md, tr["trajectories"], grade_order)
                worst = next(
                    (g for g in ("severe_collision", "moderate_collision", "mild_collision", "feasible")
                     if g in tr["trajectories"]),
                    grade_order[0],
                )
                viz.figure_collision_intervals(
                    fdir / "collision_intervals.png", occ, md, worst, tr["trajectories"][worst]
                )
                core = "moderate_collision" if "moderate_collision" in tr["trajectories"] else worst
                viz.figure_source_methods(
                    fdir / "source_methods.png", occ, md, core, tr["trajectories"][core],
                    {m: source_store[(core, m)] for m in SOURCE_METHODS}, cfg,
                )
                viz.figure_field_methods(
                    fdir / "field_methods_linear.png", occ, md, grade_order,
                    {g: tr["trajectories"][g] for g in grade_order}, cases,
                )
                viz.figure_weight_ablation(
                    fdir / "M2_weight_ablation.png", occ, md, grade_order,
                    {g: tr["trajectories"][g] for g in grade_order}, cases,
                )
                viz.figure_field_difference(
                    fdir / "field_difference.png", occ, md, grade_order,
                    {g: tr["trajectories"][g] for g in grade_order}, cases, ref_field["linear"],
                )
                viz.figure_owners(
                    fdir / "owners.png", occ, md, core, tr["trajectories"][core], cases
                )
                viz.figure_progress_profiles(
                    fdir / "progress_profiles.png", md, grade_order,
                    {g: tr["trajectories"][g] for g in grade_order},
                    {k: v for k, v in source_store.items() if k[0] in grade_order}, cases,
                )

            scene_summaries.append(
                {
                    **md,
                    "trajectory_coverage": tr["coverage"],
                    "num_trajectories": len(tr["trajectories"]),
                    "num_field_cases": scene_records,
                }
            )
            gallery.append(scene)
            print(
                f"[{sid}] {map_type} {md['type_name']}  free={md['stats']['free_fraction']:.2f} "
                f"d(S,G)={md['stats']['geodesic_sg']}  ref L={tr['reference']['length']:.0f}  "
                f"candidates={tr['coverage']['num_candidates']}  grades={len(tr['trajectories'])}  "
                f"fields={scene_records}",
                flush=True,
            )

    # ---- analysis artefacts ----
    with (root / "analysis" / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in CSV_COLUMNS})
    aggregate = _aggregate(records)
    _write_json(root / "analysis" / "aggregate_metrics.json", {"aggregate": aggregate, "config": cfg.as_dict()})

    if make_figures and records:
        viz.figure_collision_severity(root / "figures" / "summary" / "collision_severity.png", records)
        viz.figure_method_comparison(root / "figures" / "summary" / "method_comparison.png", records)
        viz.figure_field_metrics(root / "figures" / "summary" / "field_metrics.png", records)
    if make_figures and gallery:
        viz.figure_map_gallery(root / "figures" / "map_gallery.png", gallery, cfg)

    missing = [
        {
            "scene_id": s["scene_id"],
            "map_type": s["map_type"],
            "missing_grades": s["trajectory_coverage"]["missing_grades"],
        }
        for s in scene_summaries
        if s["trajectory_coverage"]["missing_grades"]
    ]
    summary = {
        "config": cfg.as_dict(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round(time.time() - t_start, 2),
        "totals": {
            "maps": len(scene_summaries),
            "trajectories": sum(s["num_trajectories"] for s in scene_summaries),
            "field_cases": len(records),
            "maps_with_all_grades": sum(1 for s in scene_summaries if not _missing_grades(s)),
        },
        "missing_collision_grades": missing,
        "aggregate": aggregate,
        "scenes": scene_summaries,
    }
    _write_json(root / "summary.json", summary)
    return summary


def _missing_grades(s: dict) -> list:
    return s["trajectory_coverage"]["missing_grades"]


def _weights_for(src: dict, kind: str) -> np.ndarray:
    from guidance_stage2.fields import source_weights

    return source_weights(np.asarray(src["progress"], dtype=np.float64), kind)
