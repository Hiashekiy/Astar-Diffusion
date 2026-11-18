"""The stage-1 mechanism: trajectory samples -> weighted multi-source propagation.

    F(x) = max_i [ w_i - lambda * d_free(x, s_i) ]

implemented as a max-Dijkstra from all trajectory samples at once, so obstacles
block propagation exactly as they block movement.
"""

from __future__ import annotations

import heapq

import numpy as np

from .config import WEIGHT_KINDS
from .grid import Grid


# --------------------------------------------------------------------------- #
# source strength
# --------------------------------------------------------------------------- #
def sample_progress(m: int) -> np.ndarray:
    """Normalised trajectory progress u_i = i / (M-1)."""
    return np.linspace(0.0, 1.0, m)


def source_weights(m: int, kind: str) -> np.ndarray:
    """w_i as a function of trajectory progress (Field-A / B / C)."""
    u = sample_progress(m)
    if kind == "uniform":
        return np.ones(m)  # Field-A: plain multi-source distance field
    if kind == "linear":
        return u.copy()  # Field-B: 0 at S -> 1 at G
    if kind == "quadratic":
        return u**2  # Field-C: reinforces the goal end
    raise ValueError(f"unknown weight kind {kind!r}, expected one of {WEIGHT_KINDS}")


def weight_at_progress(u: np.ndarray | float, kind: str) -> np.ndarray:
    """Continuous version of :func:`source_weights` (used for the ideal ramp check)."""
    u = np.asarray(u, dtype=np.float64)
    if kind == "uniform":
        return np.ones_like(u)
    if kind == "linear":
        return u.copy()
    if kind == "quadratic":
        return u**2
    raise ValueError(f"unknown weight kind {kind!r}")


# --------------------------------------------------------------------------- #
# propagation
# --------------------------------------------------------------------------- #
def propagate(
    grid: Grid, samples: np.ndarray, weights: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted multi-source propagation; returns ``(field, owner)`` on the grid.

    ``field`` is NaN on obstacles, ``owner`` holds the index of the sample that
    dominates each free cell (-1 where nothing reaches it).  Repeated sample cells
    (a trajectory that doubles back) keep their strongest weight.
    """
    size = grid.size
    field = np.full(size, -np.inf, dtype=np.float64)
    owner = np.full(size, -1, dtype=np.int32)
    flat_free = grid.free.ravel()

    seeds: list[int] = []
    for i, (r, c) in enumerate(samples):
        r, c = int(r), int(c)
        if not (0 <= r < grid.height and 0 <= c < grid.width) or not flat_free[r * grid.width + c]:
            raise ValueError(f"source {i} at {(r, c)} is not free space")
        idx = r * grid.width + c
        if weights[i] > field[idx]:
            field[idx] = float(weights[i])
            owner[idx] = i
            seeds.append(idx)

    heap = [(-field[idx], idx) for idx in set(seeds)]
    heapq.heapify(heap)
    while heap:
        neg_v, v = heapq.heappop(heap)
        v_val = -neg_v
        if v_val < field[v]:
            continue  # stale entry
        new_value = v_val - lam
        for u in grid.neighbors(v):
            u = int(u)
            if new_value > field[u]:
                field[u] = new_value
                owner[u] = owner[v]
                heapq.heappush(heap, (-new_value, u))

    field = np.where(flat_free, field, np.nan).reshape(grid.height, grid.width)
    owner = np.where(flat_free, owner, -1).reshape(grid.height, grid.width)
    return field, owner


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #
def field_stats(field: np.ndarray) -> dict:
    vals = field[np.isfinite(field)]
    if vals.size == 0:
        return {"field_min": float("nan"), "field_max": float("nan"), "field_mean": float("nan"), "field_std": float("nan")}
    return {
        "field_min": float(vals.min()),
        "field_max": float(vals.max()),
        "field_mean": float(vals.mean()),
        "field_std": float(vals.std()),
    }


def masked_mean(field: np.ndarray, mask: np.ndarray) -> float:
    vals = field[mask & np.isfinite(field)]
    return float(vals.mean()) if vals.size else float("nan")


def owner_stats(owner: np.ndarray, m: int) -> dict:
    """How many distinct samples actually own part of the map (structure diversity)."""
    labels = owner[owner >= 0]
    if labels.size == 0:
        return {"owner_distinct": 0, "owner_entropy": 0.0, "owner_dominant_fraction": 0.0}
    counts = np.bincount(labels, minlength=m).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(m)) if m > 1 else 0.0
    return {
        "owner_distinct": int((counts > 0).sum()),
        "owner_entropy": entropy,
        "owner_dominant_fraction": float(p.max()),
    }


def path_profile(field: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Field value at each of the M trajectory samples."""
    return np.array([field[int(r), int(c)] for r, c in samples], dtype=np.float64)


def directionality(profile: np.ndarray) -> dict:
    """Along-trajectory trend: monotone fraction and least-squares slope vs. u."""
    u = sample_progress(len(profile))
    finite = np.isfinite(profile)
    if finite.sum() < 3:
        return {"monotone_fraction": 0.0, "slope": 0.0, "profile_start": 0.0, "profile_goal": 0.0}
    diffs = np.diff(profile[finite])
    slope = float(np.polyfit(u[finite], profile[finite], 1)[0])
    return {
        "monotone_fraction": float((diffs >= -1e-9).mean()),
        "slope": slope,
        "profile_start": float(profile[0]),
        "profile_goal": float(profile[-1]),
    }


def path_dip(field: np.ndarray, nodes: list[int], width: int, kind: str) -> float:
    """Max shortfall of the field below the ideal continuous ramp along the path.

    The trajectory is sampled every ``L/M`` cells, so between samples the field can
    sag slightly below ``w(u)``; a large dip would mean the high-response band along
    the trajectory is broken.
    """
    total = len(nodes) - 1
    if total <= 0:
        return 0.0
    u = np.arange(len(nodes), dtype=np.float64) / total
    ideal = weight_at_progress(u, kind)
    actual = np.array([field[divmod(int(i), width)] for i in nodes])
    ok = np.isfinite(actual)
    if not ok.any():
        return float("inf")
    return float(np.max(ideal[ok] - actual[ok]))


def lipschitz_violation(grid: Grid, field: np.ndarray, lam: float) -> float:
    """Largest excess of a field step over ``lam`` on any free edge.

    Should be ~0: it proves propagation used geodesic (wall-aware) steps of cost 1.
    """
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


def measure_field(
    field: np.ndarray,
    owner: np.ndarray,
    samples: np.ndarray,
    masks: dict,
    m: int,
    kind: str,
) -> dict:
    """All per-field numbers reported in ``summary.json``."""
    profile = path_profile(field, samples)
    directional = directionality(profile)
    stats = field_stats(field)
    out = dict(stats)
    out.update(owner_stats(owner, m))
    out.update(directional)
    out["F_path"] = float(np.nanmean(profile))
    out["F_start_ball"] = masked_mean(field, masks["start_ball"])
    out["F_goal_ball"] = masked_mean(field, masks["goal_ball"])
    out["delta_F"] = out["F_goal_ball"] - out["F_start_ball"]
    out["F_pockets"] = masked_mean(field, masks["pockets"]) if masks["pockets"].any() else float("nan")
    out["pocket_contrast"] = out["F_path"] - out["F_pockets"]
    out["weight_kind"] = kind
    return out
