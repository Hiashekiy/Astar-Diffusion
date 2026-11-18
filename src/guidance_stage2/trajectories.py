"""Stage-2A trajectory generation: one collision-free reference plus graded collisions.

Collision grades are assigned from the **measured** ``collision_ratio``, never from the
name of the generator that produced the trajectory.  Every raw trajectory is a
continuous float polyline ``(N, 2)`` in ``(row, col)``: it is never snapped back to grid
cells, because that would erase the wall-crossing geometry we are studying.
"""

from __future__ import annotations

import numpy as np

from guidance_stage1.grid import Grid
from guidance_stage1.paths import astar
from guidance_stage2.collision import (
    arc_lengths,
    detect_collision_intervals,
    measure_collision_ratio,
    polyline_length,
)

EPS = 1e-9


# --------------------------------------------------------------------------- #
# reference trajectory
# --------------------------------------------------------------------------- #
def generate_reference_astar(grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> dict:
    """Collision-free reference path from the stage-1 A* (unchanged behaviour)."""
    nodes = astar(grid, grid.idx(*start), grid.idx(*goal))
    if not nodes:
        raise RuntimeError("A* found no reference path between S and G")
    pts = np.array([divmod(int(i), grid.width) for i in nodes], dtype=np.float64)
    return {
        "nodes": nodes,
        "polyline": pts,
        "length": polyline_length(pts),
        "num_nodes": len(nodes),
    }


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _dedupe(points: np.ndarray) -> np.ndarray:
    out = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - out[-1]) > 1e-9:
            out.append(p)
    return np.asarray(out, dtype=np.float64)


def bezier_curve(control_points, num_samples: int) -> np.ndarray:
    """Bezier curve through ``control_points`` via de Casteljau (any degree)."""
    control = np.asarray(control_points, dtype=np.float64)
    n = control.shape[0]
    t = np.linspace(0.0, 1.0, num_samples)[:, None, None]
    cur = np.repeat(control[None, :, :], num_samples, axis=0)
    for _ in range(n - 1):
        cur = cur[:, :-1, :] * (1.0 - t) + cur[:, 1:, :] * t
    return cur[:, 0, :]


def _apply_shortcuts(ref_pts: np.ndarray, pairs) -> np.ndarray:
    """Replace reference sub-paths between anchor pairs with straight segments."""
    out = []
    cursor = 0
    for ia, ib in sorted(pairs):
        if ia < cursor or ib <= ia:
            continue
        out.extend(ref_pts[cursor : ia + 1])
        out.append(ref_pts[ib])
        cursor = ib
    out.extend(ref_pts[cursor:])
    return _dedupe(np.asarray(out, dtype=np.float64))


# --------------------------------------------------------------------------- #
# candidate generators
# --------------------------------------------------------------------------- #
def generate_shortcut_collision_candidates(
    grid: Grid, ref_pts: np.ndarray, rng: np.random.Generator, cfg
) -> list[dict]:
    """Method 1 (primary): straight-line shortcuts that cut through obstacles."""
    n = len(ref_pts)
    if n < 8:
        return []
    out = []
    for _ in range(cfg.num_shortcut_candidates):
        k = int(rng.integers(1, 4))
        pairs = []
        cursor = 0
        for _k in range(k):
            min_gap = int(rng.integers(4, 16))
            ia = int(rng.integers(0, max(1, n - min_gap - 1)))
            max_gap = int(rng.integers(min_gap, min(n - 1 - ia, 30) + 1)) if n - 1 - ia > min_gap else min_gap
            ib = min(n - 1, ia + max(int(min_gap), int(max_gap)))
            if ib <= ia or ia < cursor:
                continue
            pairs.append((ia, ib))
            cursor = ib + 1
        if not pairs:
            continue
        path = _apply_shortcuts(ref_pts, pairs)
        if len(path) < 3:
            continue
        out.append({"raw_path": path, "generator": "shortcut", "params": {"anchors": [[int(a), int(b)] for a, b in pairs]}})

    # Graded ladder: for a few anchor starts, sweep the far anchor.  A single shortcut
    # escalates its collision ratio roughly monotonically with the anchor gap, so a fine
    # sweep fills bands that random multi-shortcut sampling leaves empty (this is how the
    # 0.14 -> 0.31 gap in some maps gets covered).
    starts = np.linspace(0, max(0, n - 13), 10).astype(int)
    for ia in starts:
        max_gap = min(n - 1 - int(ia), 120)
        for gap in range(4, max_gap + 1, 2):
            path = _apply_shortcuts(ref_pts, [(int(ia), int(ia) + gap)])
            if len(path) < 3:
                continue
            out.append(
                {
                    "raw_path": path,
                    "generator": "shortcut",
                    "params": {"anchors": [[int(ia), int(ia + gap)]], "ladder": True},
                }
            )
    return out


def generate_bezier_collision_candidates(
    grid: Grid, ref_pts: np.ndarray, rng: np.random.Generator, cfg
) -> list[dict]:
    """Method 2 (supplement): unconstrained Bezier smoothing, then collision-checked."""
    n = len(ref_pts)
    if n < 10:
        return []
    out = []
    for _ in range(cfg.num_bezier_candidates):
        k = int(rng.integers(3, 6))
        idx = np.linspace(0, n - 1, k).astype(int)
        control = [
            ref_pts[i] + rng.normal(0.0, 0.8, size=2) if 0 < j < k - 1 else ref_pts[i]
            for j, i in enumerate(idx)
        ]
        control[0] = ref_pts[0]  # keep S exact
        control[-1] = ref_pts[-1]  # keep G exact
        curve = bezier_curve(control, num_samples=max(60, 3 * n))
        curve = _dedupe(curve)
        if len(curve) < 3:
            continue
        out.append(
            {
                "raw_path": curve,
                "generator": "bezier",
                "params": {"control_points": [[round(float(v), 2) for v in c] for c in control]},
            }
        )
    return out


def _obstacle_blobs(occ: np.ndarray, min_cells: int = 6, max_cells: int = 120) -> list[list[tuple[int, int]]]:
    from collections import deque

    h, w = occ.shape
    seen = np.zeros_like(occ, dtype=bool)
    blobs = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if occ[r, c] != 1 or seen[r, c]:
                continue
            dq = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while dq:
                rr, cc = dq.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < h and 0 <= c2 < w and occ[r2, c2] == 1 and not seen[r2, c2]:
                        seen[r2, c2] = True
                        dq.append((r2, c2))
            if min_cells <= len(cells) <= max_cells:
                blobs.append(cells)
    return blobs


def generate_directed_crossing_candidates(
    grid: Grid, occ: np.ndarray, start, goal, rng: np.random.Generator, cfg
) -> list[dict]:
    """Method 3 (coverage fallback): deliberately drive through one chosen obstacle."""
    blobs = _obstacle_blobs(occ)
    if not blobs:
        return []
    s_idx, g_idx = grid.idx(*start), grid.idx(*goal)
    out = []
    order = rng.permutation(len(blobs))
    for k in order[: cfg.num_directed_candidates]:
        cells = blobs[int(k)]
        rows = np.array([c[0] for c in cells])
        cols = np.array([c[1] for c in cells])
        horizontal = (cols.max() - cols.min()) >= (rows.max() - rows.min())
        r_mid, c_mid = int(rows.mean()), int(cols.mean())
        offset = int(rng.integers(3, 6))
        if horizontal:
            a = (r_mid, int(cols.min()) - offset)
            b = (r_mid, int(cols.max()) + offset)
        else:
            a = (int(rows.min()) - offset, c_mid)
            b = (int(rows.max()) + offset, c_mid)
        if not (grid.is_free(*a) and grid.is_free(*b)):
            continue
        leg1 = astar(grid, s_idx, grid.idx(*a))
        leg2 = astar(grid, grid.idx(*b), g_idx)
        if not leg1 or not leg2:
            continue
        pts1 = np.array([divmod(int(i), grid.width) for i in leg1], dtype=np.float64)
        pts2 = np.array([divmod(int(i), grid.width) for i in leg2], dtype=np.float64)
        path = _dedupe(np.vstack([pts1, np.array([b], dtype=np.float64), pts2]))
        out.append(
            {
                "raw_path": path,
                "generator": "directed_crossing",
                "params": {"blob_cells": len(cells), "side_a": [int(a[0]), int(a[1])], "side_b": [int(b[0]), int(b[1])]},
            }
        )
    return out


# --------------------------------------------------------------------------- #
# candidate evaluation + grading
# --------------------------------------------------------------------------- #
def _evaluate(grid: Grid, raw_path: np.ndarray, ref_length: float, cfg) -> dict:
    metrics = measure_collision_ratio(grid, raw_path, step=cfg.collision_detection_step)
    length = float(metrics["path_length"])
    metrics["reference_path_length"] = float(ref_length)
    metrics["rho"] = float(length / ref_length) if ref_length > 0 else float("inf")
    return metrics


def _resample_arc(path: np.ndarray, n: int = 200) -> np.ndarray:
    from guidance_stage2.collision import point_at_arc_length

    cum = arc_lengths(path)
    s = np.linspace(0.0, cum[-1], n)
    return np.stack([point_at_arc_length(path, cum, t) for t in s])


def _synthesize_gap_candidates(
    grid: Grid, evaluated: list[dict], missing: list[dict], ref_length: float, cfg
) -> list[dict]:
    """Blend a below-band and an above-band candidate to fill a collision band.

    Only used when the primary generators leave a band empty.  Blending two trajectories
    that both run S -> G at the same parameterisation stays a continuous S -> G
    trajectory, and its collision ratio moves continuously between the two, so a bracket
    can be resolved by scanning the blend weight.
    """
    if not evaluated:
        return []
    ratios = np.array([c["metrics"]["collision_ratio"] for c in evaluated])
    out = []
    for entry in missing:
        lo, hi = entry["band"]
        centre = 0.5 * (lo + hi)
        below = np.where(ratios < lo)[0]
        above = np.where(ratios >= hi)[0]
        if len(below) == 0 or len(above) == 0:
            continue
        low = evaluated[int(below[np.argmax(ratios[below])])]
        high = evaluated[int(above[np.argmin(ratios[above])])]
        a = _resample_arc(low["raw_path"])
        b = _resample_arc(high["raw_path"])
        best = None
        for t in np.linspace(0.0, 1.0, 33):
            blended = (1.0 - t) * a + t * b
            metrics = _evaluate(grid, blended, ref_length, cfg)
            if metrics["collision_ratio"] < lo or metrics["collision_ratio"] >= hi:
                continue
            score = abs(metrics["collision_ratio"] - centre)
            if best is None or score < best[0]:
                best = (score, blended, metrics)
        if best is None:
            continue
        out.append(
            {
                "key": ("blend", entry["grade"], round(float(best[1].sum()), 6)),
                "raw_path": best[1],
                "generator": "blend",
                "params": {"from_ratios": [low["metrics"]["collision_ratio"], high["metrics"]["collision_ratio"]],
                           "target_band": [lo, hi]},
                "metrics": best[2],
            }
        )
    return out


def select_by_collision_band(candidates: list[dict], bands: dict) -> tuple[dict, list[dict]]:
    """Pick one trajectory per collision band by *measured* ratio; report what is missing."""
    chosen: dict[str, dict] = {}
    used: set = set()
    for name, (lo, hi) in bands.items():
        centre = 0.5 * (lo + hi)
        within = [
            c
            for c in candidates
            if c["key"] not in used and lo <= c["metrics"]["collision_ratio"] < hi
        ]
        if not within:
            continue
        best = min(within, key=lambda c: (abs(c["metrics"]["collision_ratio"] - centre), c["metrics"]["path_length"]))
        chosen[name] = best
        used.add(best["key"])
    missing = [
        {
            "grade": name,
            "band": [lo, hi],
            "best_available": max((c["metrics"]["collision_ratio"] for c in candidates), default=0.0),
        }
        for name, (lo, hi) in bands.items()
        if name not in chosen
    ]
    return chosen, missing


def generate_trajectories(grid: Grid, occ: np.ndarray, start, goal, cfg, rng: np.random.Generator) -> dict:
    """Reference path + G0..G3 graded trajectories + generation bookkeeping."""
    reference = generate_reference_astar(grid, start, goal)
    ref_pts = reference["polyline"]
    ref_len = reference["length"]

    candidates: list[dict] = []
    for gen in (
        generate_shortcut_collision_candidates(grid, ref_pts, rng, cfg),
        generate_bezier_collision_candidates(grid, ref_pts, rng, cfg),
        generate_directed_crossing_candidates(grid, occ, start, goal, rng, cfg),
    ):
        candidates.extend(gen)

    evaluated = []
    for cand in candidates:
        raw = np.asarray(cand["raw_path"], dtype=np.float64)
        if len(raw) < 3 or not np.all(np.isfinite(raw)):
            continue
        # a graded trajectory must start at S and end at G, exactly
        if np.linalg.norm(raw[0] - ref_pts[0]) > 1e-6 or np.linalg.norm(raw[-1] - ref_pts[-1]) > 1e-6:
            continue
        metrics = _evaluate(grid, raw, ref_len, cfg)
        evaluated.append(
            {
                "key": (round(float(raw.sum()), 6), len(raw)),
                "raw_path": raw,
                "generator": cand["generator"],
                "params": cand["params"],
                "metrics": metrics,
            }
        )

    chosen, missing = select_by_collision_band(evaluated, cfg.collision_bands())
    if missing:
        extra = _synthesize_gap_candidates(grid, evaluated, missing, ref_len, cfg)
        if extra:
            evaluated.extend(extra)
            chosen, missing = select_by_collision_band(evaluated, cfg.collision_bands())

    trajectories = {
        "feasible": {
            "raw_path": ref_pts.copy(),
            "generator": "astar_reference",
            "params": {},
            "metrics": _evaluate(grid, ref_pts, ref_len, cfg),
        }
    }
    for name, cand in chosen.items():
        trajectories[name] = {
            "raw_path": cand["raw_path"],
            "generator": cand["generator"],
            "params": cand["params"],
            "metrics": cand["metrics"],
        }

    coverage = {
        "num_candidates": len(evaluated),
        "num_by_generator": {
            g: sum(1 for c in evaluated if c["generator"] == g) for g in ("shortcut", "bezier", "directed_crossing")
        },
        "num_with_collision": sum(1 for c in evaluated if c["metrics"]["collision_ratio"] > 0),
        "max_collision_ratio": max((c["metrics"]["collision_ratio"] for c in evaluated), default=0.0),
        "missing_grades": missing,
    }
    return {"reference": reference, "trajectories": trajectories, "coverage": coverage}
