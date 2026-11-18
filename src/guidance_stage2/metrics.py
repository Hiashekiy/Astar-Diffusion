"""Quantitative indicators for stage 2A: trajectory, source handling, field, channels."""

from __future__ import annotations

import numpy as np

from guidance_stage1.grid import Grid

# --------------------------------------------------------------------------- #
# source handling
# --------------------------------------------------------------------------- #
def source_metrics(sources: dict) -> dict:
    meta = sources["metadata"]
    progress = np.asarray(sources["progress"], dtype=np.float64)
    positions = np.asarray(sources["positions_rc"]).reshape(-1, 2)
    cells = [tuple(int(v) for v in p) for p in positions]
    out = {
        "num_raw_samples": meta.get("num_raw_samples"),
        "num_valid_sources": int(len(positions)),
        "valid_source_ratio": round(float(meta.get("valid_source_ratio", 0.0)), 4),
        "num_duplicate_source_cells": int(len(cells) - len(set(cells))),
        "num_invalid_sources": int(meta.get("num_invalid_sources", 0)),
        "mean_projection_distance": meta.get("mean_projection_distance"),
        "max_projection_distance": meta.get("max_projection_distance"),
        "num_collision_intervals": meta.get("num_collision_intervals"),
        "num_bridged_intervals": meta.get("num_bridged_intervals"),
        "bridge_success_rate": meta.get("bridge_success_rate"),
        "bridge_total_length": meta.get("bridge_total_length"),
        "bridge_detour_ratio": meta.get("bridge_detour_ratio"),
        "roi_expansion_count": meta.get("roi_expansion_count"),
        "progress_min": round(float(progress.min()), 4) if len(progress) else None,
        "progress_max": round(float(progress.max()), 4) if len(progress) else None,
        "progress_covered_fraction": round(float(_covered_fraction(progress)), 4) if len(progress) else None,
        "num_failures": len(meta.get("failures", [])),
    }
    if "num_projected_sources" in meta:
        out["num_projected_sources"] = meta["num_projected_sources"]
    return out


def _covered_fraction(progress: np.ndarray, tol: float = 0.01) -> float:
    """Fraction of the [0, 1] progress axis that has a source within ``tol``.

    A dropped collision interval shows up here as a real gap: this is the quantitative
    form of "the trajectory lost its middle".
    """
    if len(progress) == 0:
        return 0.0
    grid_u = np.linspace(0.0, 1.0, 201)
    d = np.abs(grid_u[:, None] - progress[None, :]).min(axis=1)
    return float((d <= tol).mean())


# --------------------------------------------------------------------------- #
# field structure
# --------------------------------------------------------------------------- #
def field_metrics(grid: Grid, field: np.ndarray, owner: np.ndarray, sources: dict, masks: dict, lam: float) -> dict:
    free = grid.free
    finite = np.isfinite(field)
    vals = field[free & finite]
    out = {
        "field_min": float(vals.min()) if vals.size else None,
        "field_max": float(vals.max()) if vals.size else None,
        "field_mean": float(vals.mean()) if vals.size else None,
        "field_std": float(vals.std()) if vals.size else None,
        "free_cell_finite_ratio": round(float((finite & free).sum() / max(free.sum(), 1)), 6),
        "obstacle_nan_ratio": round(float(np.isnan(field[~free]).sum() / max(int((~free).sum()), 1)), 6),
        "num_sources": int(sources["metadata"].get("num_valid_sources", 0)),
    }
    labels = owner[owner >= 0]
    if labels.size:
        counts = np.bincount(labels).astype(np.float64)
        p = counts[counts > 0] / counts.sum()
        m = max(len(p), 2)
        out["owner_distinct"] = int((counts > 0).sum())
        out["owner_entropy"] = float(-(p * np.log(p)).sum() / np.log(m))
        out["owner_dominant_fraction"] = float(p.max())
    else:
        out["owner_distinct"] = 0
        out["owner_entropy"] = 0.0
        out["owner_dominant_fraction"] = 0.0

    f_start = _masked_mean(field, masks["start_ball"])
    f_goal = _masked_mean(field, masks["goal_ball"])
    out["F_start_ball"] = f_start
    out["F_goal_ball"] = f_goal
    out["delta_F"] = None if (f_start is None or f_goal is None) else float(f_goal - f_start)

    positions = np.asarray(sources["positions_rc"]).reshape(-1, 2)
    progress = np.asarray(sources["progress"], dtype=np.float64)
    if len(positions) >= 3:
        order = np.argsort(progress)
        p_sorted = progress[order]
        f_sorted = np.array([field[int(r), int(c)] for r, c in positions[order]], dtype=np.float64)
        ok = np.isfinite(f_sorted)
        if ok.sum() >= 3:
            diffs = np.diff(f_sorted[ok])
            out["monotone_fraction"] = float((diffs >= -1e-9).mean())
            out["profile_slope"] = float(np.polyfit(p_sorted[ok], f_sorted[ok], 1)[0])
        else:
            out["monotone_fraction"] = None
            out["profile_slope"] = None
    else:
        out["monotone_fraction"] = None
        out["profile_slope"] = None

    out["lipschitz_violation"] = _lipschitz_violation(grid, field, lam)
    return out


def _masked_mean(field: np.ndarray, mask: np.ndarray):
    vals = field[mask & np.isfinite(field)]
    return float(vals.mean()) if vals.size else None


def _lipschitz_violation(grid: Grid, field: np.ndarray, lam: float) -> float:
    """max(|F(u) - F(v)| - lambda) over free edges: 0 means every step obeys the decay."""
    flat = field.ravel()
    worst = 0.0
    for v in grid.free_idx:
        fv = flat[v]
        if not np.isfinite(fv):
            continue
        for u in grid.neighbors(int(v)):
            fu = flat[u]
            if np.isfinite(fu):
                worst = max(worst, abs(fv - fu) - lam)
    return float(max(worst, 0.0))


# --------------------------------------------------------------------------- #
# reference-channel preference (path independent)
# --------------------------------------------------------------------------- #
def channel_metrics(grid: Grid, field: np.ndarray, detour: np.ndarray, cfg) -> dict:
    near = grid.free & np.isfinite(detour) & (detour <= cfg.detour_near)
    far = grid.free & np.isfinite(detour) & (detour >= cfg.detour_far)
    f_near = _masked_mean(field, near)
    f_far = _masked_mean(field, far)
    out = {
        "near_channel_cells": int(near.sum()),
        "off_channel_cells": int(far.sum()),
        "F_near_channel": f_near,
        "F_off_channel": f_far,
        "channel_gap": None if (f_near is None or f_far is None) else float(f_near - f_far),
        "channel_auc": _auc(field[near], field[far]),
    }
    return out


def _auc(positive: np.ndarray, negative: np.ndarray):
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        return None
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([positive, negative]))
    n1, n0 = positive.size, negative.size
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def compare_to_reference(grid: Grid, field: np.ndarray, ref_field: np.ndarray, ref_mask: np.ndarray) -> dict:
    """Agreement with the field built from the collision-free reference trajectory."""
    ok = grid.free & np.isfinite(field) & np.isfinite(ref_field)
    if ok.sum() < 10:
        return {
            "mae_all_free": None,
            "corr_all_free": None,
            "mae_near_reference_path": None,
            "channel_mean_diff": None,
            "high_response_coverage": None,
        }
    a, b = field[ok], ref_field[ok]
    near = ok & ref_mask
    ref_max = float(np.nanmax(ref_field))
    threshold = ref_max - 0.1
    cov = float((field[near] >= threshold).mean()) if near.sum() else None
    return {
        "mae_all_free": float(np.mean(np.abs(a - b))),
        "corr_all_free": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None,
        "mae_near_reference_path": float(np.mean(np.abs(field[near] - ref_field[near]))) if near.sum() else None,
        "channel_mean_diff": float(np.mean(field[near] - ref_field[near])) if near.sum() else None,
        "high_response_coverage": cov,
    }


def corridor_mask(grid: Grid, nodes, radius: int = 2) -> np.ndarray:
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    for i in nodes:
        r, c = divmod(int(i), grid.width)
        mask[max(0, r - radius) : r + radius + 1, max(0, c - radius) : c + radius + 1] = True
    return mask & grid.free
