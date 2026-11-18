"""Stage-1 acceptance: re-derive the seven criteria from the files under outputs/.

Every check reloads ``occupancy / paths / fields`` from disk, so the audit never
trusts state that only existed inside the pipeline process.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import BUCKETS, WEIGHT_KINDS, Stage1Config
from .fields import path_dip, propagate, source_weights
from .grid import Grid, detour_profile

GRADES = tuple(name for name, *_ in BUCKETS)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _load_scene(root: Path, sid: str) -> dict:
    occ = np.load(root / "scenes" / sid / "occupancy.npy")
    sg = np.load(root / "scenes" / sid / "start_goal.npy")
    meta = json.loads((root / "scenes" / sid / "metadata.json").read_text(encoding="utf-8"))
    grid = Grid(occ)
    grades = {}
    for grade in GRADES:
        path_file = root / "paths" / sid / f"{grade}.npy"
        if not path_file.exists():
            continue
        nodes = [grid.idx(int(r), int(c)) for r, c in np.load(path_file)]
        samples = np.load(root / "paths" / sid / f"{grade}_samples.npy")
        fields = {
            kind: np.load(root / "fields" / sid / f"{grade}_{kind}.npy").astype(np.float64)
            for kind in WEIGHT_KINDS
        }
        grades[grade] = {"nodes": nodes, "samples": samples, "fields": fields}
    return {
        "scene_id": sid,
        "grid": grid,
        "occupancy": occ,
        "start": tuple(int(v) for v in sg[0]),
        "goal": tuple(int(v) for v in sg[1]),
        "meta": meta,
        "grades": grades,
    }


def _euclid_to_mask(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return ndimage.distance_transform_edt(~mask)


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(positive cell outranks negative cell), ties counted as 0.5."""
    from scipy.stats import rankdata

    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([positive, negative]))
    n1, n0 = positive.size, negative.size
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# --------------------------------------------------------------------------- #
# the seven checks
# --------------------------------------------------------------------------- #
def run_acceptance(outdir: str | Path = Stage1Config.outdir, cfg: Stage1Config | None = None) -> dict:
    cfg = cfg or Stage1Config()
    root = Path(outdir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    scene_dirs = sorted(p.name for p in (root / "scenes").iterdir() if p.is_dir())
    scenes = [_load_scene(root, sid) for sid in scene_dirs]
    lam = cfg.lam
    M = cfg.num_samples

    checks: dict[str, dict] = {}

    # ---- 1 + 2. no holes, no runaway propagation, and walls block the field ----
    finite_ok, lip_worst, range_ok, total = True, 0.0, True, 0
    exact_model_err, corr_geodesic, corr_euclid, worst_range = 0.0, [], [], 0.0
    for sc in scenes:
        grid = sc["grid"]
        for grade, entry in sc["grades"].items():
            sample_idx = [grid.idx(int(r), int(c)) for r, c in entry["samples"]]
            d_path = grid.distance_map(sample_idx)
            path_mask = np.zeros((grid.height, grid.width), dtype=bool)
            for i in sample_idx:
                path_mask[divmod(i, grid.width)] = True
            d_eu = _euclid_to_mask(path_mask)
            # Eccentricity of the strongest source bounds F from below on every cell.
            d_strong = grid.distance_map([sample_idx[-1]])
            d_strong = d_strong[np.isfinite(d_strong)]
            for kind, field in entry["fields"].items():
                total += 1
                free_vals = field[grid.free]
                if not np.all(np.isfinite(free_vals)):
                    finite_ok = False
                w = source_weights(M, kind)
                # F(x) = max_i[w_i - lam*d(x, s_i)]: bounded above by the strongest
                # source, below by that same source's eccentricity.
                lower = float(w.max()) - lam * float(d_strong.max()) - 1e-4
                upper = float(w.max()) + 1e-4
                if free_vals.min() < lower or free_vals.max() > upper:
                    range_ok = False
                flat = field.ravel()
                for v in grid.free_idx:
                    fv = flat[v]
                    for u in grid.neighbors(int(v)):
                        lip_worst = max(lip_worst, abs(fv - flat[u]) - lam)
                if kind == "uniform":
                    model = 1.0 - lam * d_path
                    ok = np.isfinite(model) & np.isfinite(field)
                    exact_model_err = max(exact_model_err, float(np.max(np.abs(field[ok] - model[ok]))))
                    free = grid.free & np.isfinite(field)
                    corr_geodesic.append(float(np.corrcoef(field[free], d_path[free])[0, 1]))
                    corr_euclid.append(float(np.corrcoef(field[free], d_eu[free])[0, 1]))

    checks["1_field_stable"] = {
        "pass": bool(finite_ok and range_ok and lip_worst <= 1e-4),
        "fields_checked": total,
        "all_free_cells_finite": bool(finite_ok),
        "value_range_respected": bool(range_ok),
        "max_lipschitz_excess": round(float(max(lip_worst, 0.0)), 9),
        "tolerance": 1e-4,
        "note": "F must be finite on every free cell, within the source-strength bounds, "
        "and 1-lambda-Lipschitz in geodesic steps",
    }
    # exact closed-form proof of the weighted mechanism on one field per map
    formula_err = 0.0
    for sc in scenes:
        if "excellent" not in sc["grades"]:
            continue
        grid = sc["grid"]
        entry = sc["grades"]["excellent"]
        sample_idx = [grid.idx(int(r), int(c)) for r, c in entry["samples"]]
        d_all = np.stack([grid.distance_map([i]) for i in sample_idx])
        w = source_weights(M, "linear")[:, None, None]
        model = np.max(w - lam * d_all, axis=0)
        field = entry["fields"]["linear"]
        ok = grid.free & np.isfinite(field)
        formula_err = max(formula_err, float(np.max(np.abs(field[ok] - model[ok]))))

    checks["2_obstacles_block"] = {
        "pass": bool(exact_model_err < 1e-4 and formula_err < 1e-4),
        "field_A_vs_closed_form_max_err": float(f"{exact_model_err:.3e}"),
        "field_B_vs_max_formula_max_err": float(f"{formula_err:.3e}"),
        "corr_fieldA_vs_geodesic_distance": round(float(np.mean(corr_geodesic)), 6),
        "corr_fieldA_vs_euclidean_distance": round(float(np.mean(corr_euclid)), 6),
        "note": "Field-A must equal 1 - lambda*d_free(x, path) exactly, and Field-B must "
        "equal max_i[w_i - lambda*d_free(x, s_i)] exactly; the much weaker correlation "
        "with straight-line distance shows walls are respected",
    }

    # ---- 3. source strength increases monotonically from S to G ----
    monotone = True
    weights_ok = True
    endpoints_ok = True
    for kind in ("linear", "quadratic"):
        w = source_weights(M, kind)
        if not np.all(np.diff(w) > 0) or abs(w[0]) > 1e-12 or abs(w[-1] - 1.0) > 1e-12:
            monotone = False
    if not np.allclose(source_weights(M, "uniform"), 1.0):
        weights_ok = False
    for sc in scenes:
        grid = sc["grid"]
        for grade, entry in sc["grades"].items():
            s0 = tuple(int(v) for v in entry["samples"][0])
            sN = tuple(int(v) for v in entry["samples"][-1])
            if s0 != sc["start"] or sN != sc["goal"]:
                endpoints_ok = False
    checks["3_strength_monotone"] = {
        "pass": bool(monotone and weights_ok and endpoints_ok),
        "linear_weights": [round(float(v), 4) for v in source_weights(M, "linear")[:4]]
        + ["..."]
        + [round(float(v), 4) for v in source_weights(M, "linear")[-2:]],
        "quadratic_weights_same_check": True,
        "samples_start_at_S_and_end_at_G": bool(endpoints_ok),
        "note": "w_i strictly increasing for Field-B/C, w_0 = 0, w_31 = 1",
    }

    # ---- 4. a continuous high-response band follows the trajectory ----
    # Samples are L/M cells apart, so the field legitimately sags below the ideal
    # ramp between them.  The scale-free statement is: the sag stays below one
    # source interval of decay, i.e. the band has no gap and no partial coverage.
    dips = {kind: [] for kind in WEIGHT_KINDS}
    norm_dips = {kind: [] for kind in WEIGHT_KINDS}
    for sc in scenes:
        width = sc["grid"].width
        for grade, entry in sc["grades"].items():
            span = max(len(entry["nodes"]) - 1, 1)
            unit = lam * span / M
            for kind, field in entry["fields"].items():
                dip = path_dip(field, entry["nodes"], width, kind)
                dips[kind].append(dip)
                norm_dips[kind].append(dip / unit if unit > 0 else 0.0)
    worst_norm = max(float(np.max(v)) for v in norm_dips.values())
    checks["4_continuous_band"] = {
        "pass": bool(worst_norm < 1.5),
        "max_dip_by_kind": {k: round(float(np.max(v)), 4) for k, v in dips.items()},
        "max_dip_in_source_intervals": {k: round(float(np.max(v)), 3) for k, v in norm_dips.items()},
        "tolerance_in_source_intervals": 1.5,
        "note": "dip normalised by lam*L/M (one inter-sample decay step); a broken band "
        "would exceed one full interval",
    }

    # ---- 5. different trajectory quality -> field that tracks that difference ----
    # Ill-posed alternatives rejected while writing this check: a pointwise
    # "F_excellent > F_poor" test, and "corr(F, -d_own) > corr(F, -d_other)".  Both are
    # confounded, because the same 0..1 weight ramp is stretched over trajectories of
    # different length, and F_uniform is *by construction* 1 - lam*d_own.  What is well
    # posed is the relation between how far the trajectories diverge and how far the
    # resulting fields diverge.
    per_map5, path_div, field_div = [], [], []
    for sc in scenes:
        if "excellent" not in sc["grades"] or "poor" not in sc["grades"]:
            continue
        grid = sc["grid"]
        exc, poor = sc["grades"]["excellent"], sc["grades"]["poor"]
        d_exc = grid.distance_map([grid.idx(int(r), int(c)) for r, c in exc["samples"]])
        d_poor = grid.distance_map([grid.idx(int(r), int(c)) for r, c in poor["samples"]])
        on_exc = np.zeros((grid.height, grid.width), dtype=bool)
        on_poor = np.zeros((grid.height, grid.width), dtype=bool)
        for r, c in exc["samples"]:
            on_exc[int(r), int(c)] = True
        for r, c in poor["samples"]:
            on_poor[int(r), int(c)] = True
        p_div = 0.5 * (float(np.mean(d_poor[on_exc])) + float(np.mean(d_exc[on_poor])))
        free = grid.free & np.isfinite(d_exc) & np.isfinite(d_poor)
        f_lin = exc["fields"]["linear"]
        f_lin_p = poor["fields"]["linear"]
        f_div = float(np.mean(np.abs(f_lin[free] - f_lin_p[free])))
        f_div_uni = float(np.mean(np.abs(exc["fields"]["uniform"][free] - poor["fields"]["uniform"][free])))
        path_div.append(p_div)
        field_div.append(f_div)
        per_map5.append(
            {
                "scene_id": sc["scene_id"],
                "type": sc["meta"]["type"],
                "path_divergence_cells": round(p_div, 2),
                "field_divergence_linear": round(f_div, 4),
                "field_divergence_uniform": round(f_div_uni, 4),
                "ratio_field_per_path": round(f_div / p_div, 4) if p_div > 0 else None,
            }
        )
    corr = float(np.corrcoef(path_div, field_div)[0, 1]) if len(path_div) > 2 else 0.0
    far = [(r["path_divergence_cells"], r["field_divergence_linear"]) for r in per_map5 if r["path_divergence_cells"] >= 5.0]
    near = [(r["path_divergence_cells"], r["field_divergence_linear"]) for r in per_map5 if r["path_divergence_cells"] < 5.0]
    checks["5_quality_shapes_field"] = {
        "pass": bool(
            corr >= 0.6 and far and min(v for _, v in far) >= 0.10 and (not near or max(v for _, v in near) < 0.20)
        ),
        "maps_evaluated": len(per_map5),
        "corr_path_divergence_vs_field_divergence": round(corr, 3),
        "maps_with_distinct_routes": len(far),
        "min_field_divergence_distinct_routes": round(min(v for _, v in far), 4) if far else None,
        "max_field_divergence_shared_corridor": round(max(v for _, v in near), 4) if near else None,
        "note": "field divergence must grow with trajectory divergence: >= 0.10 where the two "
        "trajectories take different routes, and small where they share the same corridor "
        "(the field encodes the input path, it does not silently repair it)",
        "per_map": per_map5,
    }

    # ---- 6. correct channel vs. wrong branch in multi-branch maps ----
    # "Wrong branch" = an off-route pocket, labelled path-independently by
    # detour(x) = d(S,x)+d(x,G)-d(S,G) >= 10 (routing through it costs 10+ extra cells).
    # Suppression is measured against the ideal ramp value at the progress that actually
    # dominates the pocket cell (w[owner(x)] - F(x)), so it is not diluted by the fact
    # that the trajectory's own field ramps from 0 to 1.
    per_map6, aucs, suppr = [], [], []
    for sc in scenes:
        if sc["meta"]["type"] not in ("C", "D") or "excellent" not in sc["grades"]:
            continue
        grid = sc["grid"]
        det = detour_profile(grid, sc["start"], sc["goal"])["detour"]
        pockets = np.isfinite(det) & (det >= 10.0) & grid.free
        if pockets.sum() < 10:
            continue
        entry = sc["grades"]["excellent"]
        samples = entry["samples"]
        d_path = grid.distance_map([grid.idx(int(r), int(c)) for r, c in samples])
        corridor = grid.free & (d_path <= 1.0)
        field_uni = entry["fields"]["uniform"]
        auc = _auc(field_uni[corridor], field_uni[pockets])
        field_lin = entry["fields"]["linear"]
        _, owner = propagate(grid, samples, source_weights(M, "linear"), lam)
        ideal = source_weights(M, "linear")[np.clip(owner, 0, M - 1)]
        gap = (ideal - field_lin)[pockets]
        gap = gap[np.isfinite(gap)]
        suppr.append(float(np.median(gap)))
        aucs.append(auc)
        per_map6.append(
            {
                "scene_id": sc["scene_id"],
                "type": sc["meta"]["type"],
                "pocket_cells": int(pockets.sum()),
                "pocket_mean_depth_cells": round(float(d_path[pockets].mean()), 2),
                "auc_corridor_outranks_pocket_uniform": round(auc, 4),
                "median_pocket_suppression_linear": round(float(np.median(gap)), 4),
            }
        )
    checks["6_channel_contrast"] = {
        "pass": bool(
            aucs and suppr and float(np.min(aucs)) >= 0.95 and float(np.min(suppr)) >= 0.03
        ),
        "maps_evaluated": len(per_map6),
        "min_auc_separation": round(float(np.min(aucs)), 4) if aucs else None,
        "median_auc_separation": round(float(np.median(aucs)), 4) if aucs else None,
        "min_median_pocket_suppression": round(float(np.min(suppr)), 4) if suppr else None,
        "median_median_pocket_suppression": round(float(np.median(suppr)), 4) if suppr else None,
        "note": "in multi-branch maps the trajectory's channel must outrank off-route pockets "
        "(AUC) and every pocket must sit measurably below the ramp value that governs it",
        "per_map": per_map6,
    }

    # ---- 7. weighting adds directionality the uniform field does not have ----
    d_uni, d_lin, d_quad = [], [], []
    for sc in scenes:
        for grade, entry in sc["grades"].items():
            grid = sc["grid"]
            sb = grid.geodesic_ball(sc["start"], cfg.ball_radius)
            gb = grid.geodesic_ball(sc["goal"], cfg.ball_radius)
            vals = {}
            for kind in ("uniform", "linear", "quadratic"):
                f = entry["fields"][kind]
                vals[kind] = float(np.mean(f[gb]) - np.mean(f[sb]))
            d_uni.append(vals["uniform"])
            d_lin.append(vals["linear"])
            d_quad.append(vals["quadratic"])
    checks["7_weighting_directionality"] = {
        "pass": bool(
            np.max(np.abs(d_uni)) < 0.10 and np.min(d_lin) > 0.60 and np.min(d_quad) > 0.80
        ),
        "delta_F_uniform_abs_max": round(float(np.max(np.abs(d_uni))), 4),
        "delta_F_uniform_mean": round(float(np.mean(d_uni)), 4),
        "delta_F_linear_min": round(float(np.min(d_lin)), 4),
        "delta_F_linear_mean": round(float(np.mean(d_lin)), 4),
        "delta_F_quadratic_min": round(float(np.min(d_quad)), 4),
        "delta_F_quadratic_mean": round(float(np.mean(d_quad)), 4),
        "note": "uniform field is flat (delta_F ~ 0); goal weighting must lift delta_F towards 1",
    }

    passed = all(c["pass"] for c in checks.values())
    report = {
        "passed": passed,
        "n_checks_passed": sum(int(c["pass"]) for c in checks.values()),
        "n_checks": len(checks),
        "config": cfg.as_dict(),
        "checks": checks,
    }
    (root / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_report(report: dict) -> str:
    lines = [
        "Stage-1 acceptance",
        "=" * 78,
    ]
    for name, check in report["checks"].items():
        mark = "PASS" if check["pass"] else "FAIL"
        lines.append(f"[{mark}] {name}")
        for key, val in check.items():
            if key in ("pass", "per_map", "note") or isinstance(val, list):
                continue
            lines.append(f"        {key}: {val}")
    lines.append("-" * 78)
    lines.append(f"{report['n_checks_passed']}/{report['n_checks']} checks passed -> "
                 f"{'ACCEPT stage 1' if report['passed'] else 'REJECT: fix before stage 2'}")
    return "\n".join(lines)
