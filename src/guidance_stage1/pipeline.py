"""Stage-1 orchestration: scenes -> trajectories -> fields -> figures -> summary."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import viz
from .config import BUCKETS, SCENE_TYPES, WEIGHT_KINDS, Stage1Config
from .fields import (
    lipschitz_violation,
    measure_field,
    path_dip,
    propagate,
    source_weights,
)
from .grid import Grid, detour_profile
from .paths import build_candidate_pool, resample_arc_length, select_by_grade
from .scenes import generate_scene

GRADES = tuple(name for name, *_ in BUCKETS)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _scene_masks(grid: Grid, meta: dict, detour: np.ndarray, cfg) -> dict:
    pockets = np.isfinite(detour) & (detour >= 10.0) & grid.free
    return {
        "start_ball": grid.geodesic_ball(tuple(meta["start"]), cfg.ball_radius),
        "goal_ball": grid.geodesic_ball(tuple(meta["goal"]), cfg.ball_radius),
        "pockets": pockets,
    }


def _aggregate(records: list[dict]) -> dict:
    keys = (
        "delta_F",
        "F_path",
        "F_start_ball",
        "F_goal_ball",
        "F_pockets",
        "pocket_contrast",
        "owner_entropy",
        "owner_distinct",
        "monotone_fraction",
        "slope",
        "field_min",
        "field_max",
        "field_mean",
        "field_std",
        "path_dip",
        "lipschitz",
    )
    out: dict[str, dict] = {}
    for grade in GRADES:
        for kind in WEIGHT_KINDS:
            subset = [r["metrics"][kind] for r in records if r["grade"] == grade]
            if not subset:
                continue
            bucket = {}
            for key in keys:
                vals = np.array([float(s[key]) for s in subset if key in s and s[key] is not None])
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                bucket[key] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
            out[f"{grade}_{kind}"] = bucket
    return out


def run(cfg: Stage1Config, make_figures: bool = True, lambda_sweep: bool = True) -> dict:
    t_start = time.time()
    root = Path(cfg.outdir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    dirs = {name: root / name for name in ("scenes", "paths", "fields", "figures", "analysis")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    scene_summaries: list[dict] = []
    gallery: list[dict] = []
    sweep_samples: list[dict] = []

    for t_i, scene_type in enumerate(SCENE_TYPES):
        for k in range(cfg.instances_per_type):
            scene_index = t_i * cfg.instances_per_type + k
            seed = cfg.seed + 1000 * t_i + k
            scene = generate_scene(scene_index, scene_type, seed, cfg)
            meta = scene["metadata"]
            occ = scene["occupancy"]
            grid = Grid(occ)
            start = tuple(meta["start"])
            goal = tuple(meta["goal"])
            sid = meta["scene_id"]

            (dirs["scenes"] / sid).mkdir(parents=True, exist_ok=True)
            (dirs["paths"] / sid).mkdir(parents=True, exist_ok=True)
            (dirs["fields"] / sid).mkdir(parents=True, exist_ok=True)
            if make_figures:
                (dirs["figures"] / sid).mkdir(parents=True, exist_ok=True)

            np.save(dirs["scenes"] / sid / "occupancy.npy", occ)
            np.save(dirs["scenes"] / sid / "start_goal.npy", np.array([start, goal], dtype=np.int64))
            (dirs["scenes"] / sid / "metadata.json").write_text(
                json.dumps(_jsonable(meta), indent=2), encoding="utf-8"
            )

            profile = detour_profile(grid, start, goal)
            masks = _scene_masks(grid, meta, profile["detour"], cfg)
            best_len = profile["d_sg"]
            rng = np.random.default_rng(seed + 977)
            pool = build_candidate_pool(grid, start, goal, best_len, profile["detour"], cfg, rng)
            selection = select_by_grade(pool)
            missing = [g for g in GRADES if g not in selection]
            meta["stats"]["pool_size"] = len(pool)
            meta["stats"]["rho_max_in_pool"] = round(float(max(c.ratio for c in pool)), 3)
            meta["stats"]["grade_complete"] = not missing
            meta["stats"]["missing_grades"] = missing

            per_grade: dict[str, dict] = {}
            paths_meta: dict[str, dict] = {}
            for grade in GRADES:
                if grade not in selection:
                    continue
                cand = selection[grade]
                nodes = cand["nodes"]
                samples = resample_arc_length(nodes, grid.width, cfg.num_samples)
                np.save(dirs["paths"] / sid / f"{grade}.npy", np.array([divmod(i, grid.width) for i in nodes], dtype=np.int64))
                np.save(dirs["paths"] / sid / f"{grade}_samples.npy", samples)
                weights = {kind: source_weights(cfg.num_samples, kind) for kind in WEIGHT_KINDS}

                fields, owners, metrics, profiles = {}, {}, {}, {}
                for kind in WEIGHT_KINDS:
                    field, owner = propagate(grid, samples, weights[kind], cfg.lam)
                    np.save(dirs["fields"] / sid / f"{grade}_{kind}.npy", field.astype(np.float32))
                    m = measure_field(field, owner, samples, masks, cfg.num_samples, kind)
                    m["path_dip"] = path_dip(field, nodes, grid.width, kind)
                    m["lipschitz"] = lipschitz_violation(grid, field, cfg.lam)
                    m["unique_sources"] = int(len({tuple(row) for row in samples}))
                    fields[kind], owners[kind], metrics[kind] = field, owner, m
                    profiles[kind] = m
                per_grade[grade] = {
                    "cand": cand,
                    "samples": samples,
                    "weights": weights,
                    "fields": fields,
                    "owners": owners,
                    "metrics": metrics,
                    "profiles": {kind: np.array([fields[kind][r, c] for r, c in samples]) for kind in WEIGHT_KINDS},
                }
                paths_meta[grade] = {
                    "label": cand["label"],
                    "kind": cand["kind"],
                    "ratio": round(float(cand["ratio"]), 4),
                    "length": cand["length"],
                    "in_grade": bool(cand["in_grade"]),
                    "backtracks": cand["backtracks"],
                    "backtrack_fraction": round(float(cand["backtrack_fraction"]), 4),
                    "num_waypoints": cand["num_waypoints"],
                    "unique_sources": metrics["linear"]["unique_sources"],
                }
                records.append({"scene_id": sid, "type": scene_type, "grade": grade, "metrics": metrics})

                if lambda_sweep and grade in ("excellent", "poor"):
                    for lam in (0.005, 0.01, 0.02, 0.04, 0.08):
                        f_lin, o_lin = propagate(grid, samples, weights["linear"], lam)
                        f_uni, o_uni = propagate(grid, samples, weights["uniform"], lam)
                        cu = {"start_ball": masks["start_ball"], "goal_ball": masks["goal_ball"], "pockets": masks["pockets"]}
                        m_lin = measure_field(f_lin, o_lin, samples, cu, cfg.num_samples, "linear")
                        m_uni = measure_field(f_uni, o_uni, samples, cu, cfg.num_samples, "uniform")
                        sweep_samples.append(
                            {
                                "scene_id": sid,
                                "type": scene_type,
                                "grade": grade,
                                "lam": lam,
                                "lam_times_L": lam * cand["length"],
                                "delta_F_linear": m_lin["delta_F"],
                                "delta_F_uniform": m_uni["delta_F"],
                                "owner_entropy_linear": m_lin["owner_entropy"],
                                "owner_distinct_linear": m_lin["owner_distinct"],
                                "pocket_contrast_linear": m_lin["pocket_contrast"],
                            }
                        )

            (dirs["paths"] / sid / "metadata.json").write_text(
                json.dumps(_jsonable(paths_meta), indent=2), encoding="utf-8"
            )

            if make_figures and per_grade:
                display_ranges = {
                    kind: viz.weight_display_range(kind, cfg.num_samples) for kind in WEIGHT_KINDS
                }
                for grade, entry in per_grade.items():
                    viz.figure_trajectory(
                        dirs["figures"] / sid / f"{grade}.png",
                        meta,
                        occ,
                        grade,
                        entry["cand"],
                        entry["samples"],
                        entry["weights"],
                        entry["fields"],
                        cfg,
                        display_ranges=display_ranges,
                    )
                viz.figure_quality_comparison(
                    dirs["figures"] / sid / "comparison.png", meta, occ, GRADES, per_grade, cfg
                )
                viz.figure_owners(dirs["figures"] / sid / "owners.png", meta, occ, GRADES, per_grade, cfg)
                viz.figure_trajectories(
                    dirs["figures"] / sid / "trajectories.png", meta, occ, GRADES, per_grade, cfg
                )
                viz.figure_profiles(dirs["figures"] / sid / "profiles.png", meta, GRADES, per_grade, cfg)

            scene_summaries.append(
                {
                    **meta,
                    "paths": paths_meta,
                    "fields": {
                        grade: {kind: per_grade[grade]["metrics"][kind] for kind in WEIGHT_KINDS}
                        for grade in per_grade
                    },
                }
            )
            gallery.append(
                {
                    "meta": meta,
                    "occupancy": occ,
                    "per_grade": {g: {"cand": e["cand"]} for g, e in per_grade.items()},
                }
            )
            print(
                f"[{sid}] type {scene_type}  L*={best_len:.0f}  pool={len(pool)}  "
                f"grades: " + " ".join(f"{g[:3]}={paths_meta[g]['ratio']:.2f}" for g in paths_meta),
                flush=True,
            )

    aggregate = _aggregate(records)
    if make_figures and records:
        viz.figure_summary(root / "figures" / "summary.png", records)
    if make_figures and gallery:
        viz.figure_scene_gallery(root / "figures" / "scene_gallery.png", gallery, cfg)

    summary = {
        "config": cfg.as_dict(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round(time.time() - t_start, 2),
        "totals": {
            "maps": len(scene_summaries),
            "trajectories": sum(len(s["paths"]) for s in scene_summaries),
            "fields": sum(len(v) for s in scene_summaries for v in s["fields"].values()),
            "maps_with_all_grades": sum(1 for s in scene_summaries if s["stats"]["grade_complete"]),
            "grades_in_band": sum(1 for s in scene_summaries for p in s["paths"].values() if p["in_grade"]),
        },
        "aggregate": aggregate,
        "scenes": scene_summaries,
    }
    (root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    if lambda_sweep and sweep_samples:
        _write_lambda_sweep(root / "analysis", sweep_samples, make_figures)

    return summary


def _write_lambda_sweep(outdir: Path, samples: list[dict], make_figures: bool) -> None:
    """Propagation decay is the one free parameter of the mechanism: report its effect."""
    by_lam: dict[float, list[dict]] = {}
    for rec in samples:
        by_lam.setdefault(rec["lam"], []).append(rec)
    table = {}
    for lam, recs in sorted(by_lam.items()):
        table[f"{lam:g}"] = {
            "delta_F_linear_mean": float(np.mean([r["delta_F_linear"] for r in recs])),
            "delta_F_uniform_abs_max": float(np.max(np.abs([r["delta_F_uniform"] for r in recs]))),
            "owner_entropy_linear_mean": float(np.mean([r["owner_entropy_linear"] for r in recs])),
            "owner_distinct_linear_mean": float(np.mean([r["owner_distinct_linear"] for r in recs])),
            "pocket_contrast_linear_mean": float(np.mean([r["pocket_contrast_linear"] for r in recs])),
            "n": len(recs),
        }
    (outdir / "lambda_sensitivity.json").write_text(
        json.dumps({"by_lambda": table, "samples": samples}, indent=2), encoding="utf-8"
    )
    if not make_figures:
        return
    import matplotlib.pyplot as plt

    lams = sorted(by_lam)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=110)
    panels = (
        ("delta_F_linear", "delta_F of Field-B (linear weights)"),
        ("owner_entropy_linear", "normalised owner entropy"),
        ("pocket_contrast_linear", "F_path - F_pockets"),
    )
    for ax, (key, label) in zip(axes, panels):
        for grade, color in (("excellent", "#4c72b0"), ("poor", "#c44e52")):
            means = [
                float(np.mean([r[key] for r in by_lam[lam] if r["grade"] == grade])) for lam in lams
            ]
            ax.plot(lams, means, "o-", color=color, label=grade)
        ax.set_xscale("log")
        ax.set_xlabel("lambda (decay per cell)")
        ax.set_title(label, fontsize=9.5)
        ax.grid(alpha=0.3)
    axes[0].set_yscale("symlog", linthresh=1e-3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Sensitivity to the propagation decay (20 maps, excellent vs poor trajectories)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outdir.parent / "figures" / "lambda_sensitivity.png", facecolor="white")
    plt.close(fig)
