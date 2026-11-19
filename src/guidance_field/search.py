"""Search: plain A* plus the field-guided best-first family.

The guided priority is a single expression with two knobs:

    J(n) = alpha(n) * d_hat_G(n) ** gamma - (1 - alpha(n)) * F_hat(n)

* ``alpha = 0``            -> pure field follower            (field-only)
* ``alpha = 1, gamma = 1`` -> pure distance greedy on d_hat_G
* ``0 < alpha < 1``        -> linear fusion
* ``gamma != 1``           -> nonlinear fusion (gamma < 1 sharpens near the goal)
* ``adaptive``             -> alpha varies with the remaining distance

``F_hat`` and ``d_hat_G`` are normalised *before* the search, and the priority function only
ever sees those precomputed arrays, never the accumulated cost ``g``. Only ``astar`` claims
shortest-path optimality.
"""

from __future__ import annotations

import heapq
import time

import numpy as np

from .grid import Grid

# canonical display/normalisation ranges per weighting
FIELD_CLIP = {
    "uniform": (0.50, 1.00),
    "linear": (-0.10, 1.00),
    "quadratic": (-0.10, 1.00),
}


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def normalise_field(field: np.ndarray, clip=None, weight: str = "linear") -> np.ndarray:
    """Map the field into [0, 1] with a fixed clip so one alpha means one thing."""
    lo, hi = clip if clip is not None else FIELD_CLIP.get(weight, (-0.1, 1.0))
    values = np.where(np.isfinite(field), np.clip(field, lo, hi), lo)
    return (values - lo) / (hi - lo)


def normalise_distance(grid: Grid, goal) -> tuple[np.ndarray, float]:
    """d_hat_G in [0, 1]: geodesic distance to the goal divided by its maximum."""
    dist = grid.bfs_distance([grid.idx(*goal)]).astype(np.float64)
    dist[dist < 0] = np.nan
    d_max = float(np.nanmax(dist)) if np.isfinite(dist).any() else 1.0
    d_max = max(d_max, 1.0)
    out = np.where(np.isfinite(dist), dist / d_max, 1.0)
    return out, d_max


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pack(grid: Grid, path: list[int]) -> np.ndarray:
    if not path:
        return np.zeros((0, 2), dtype=int)
    rows, cols = np.divmod(np.asarray(path, dtype=int), grid.width)
    return np.stack([rows, cols], axis=1)


def path_is_valid(grid: Grid, path, start, goal) -> dict:
    """Continuity, obstacle and endpoint check on a returned path."""
    arr = np.asarray(path, dtype=int)
    if arr.size == 0:
        return {"valid": False, "reason": "empty path"}
    if len(arr) == 1:
        return {"valid": arr[0, 0] == start[0] and arr[0, 1] == goal[0], "length": 0}
    if tuple(arr[0]) != tuple(start) or tuple(arr[-1]) != tuple(goal):
        return {"valid": False, "reason": "endpoints do not match start/goal"}
    steps = np.abs(np.diff(arr, axis=0)).sum(axis=1)
    if np.any(steps != 1):
        return {"valid": False, "reason": "path is not 4-connected"}
    if not np.all(grid.free[arr[:, 0], arr[:, 1]]):
        return {"valid": False, "reason": "path crosses an obstacle"}
    return {"valid": True, "length": int(len(arr) - 1)}


def _best_first(grid: Grid, start, goal, priority, cutoff: int = 0) -> dict:
    """Generic best-first loop driven by a pure ``priority(r, c)`` callable."""
    t0 = time.perf_counter()
    s, t = grid.idx(*start), grid.idx(*goal)
    if not grid.free.flat[s] or not grid.free.flat[t]:
        return {"status": "invalid", "path": np.zeros((0, 2), int), "trace": np.zeros((0, 2), int),
                "expanded": 0, "generated": 0, "length": None, "time_ms": 0.0}
    parent = {s: -1}
    seen = {s}
    counter = 0
    heap = [(priority(*start), counter, s)]
    trace: list[int] = []
    nbr = grid._nbr
    rows_cache: dict[int, tuple[int, int]] = {}
    while heap:
        _, _, v = heapq.heappop(heap)
        trace.append(v)
        if v == t:
            path = []
            while v != -1:
                path.append(v)
                v = parent[v]
            path.reverse()
            return {
                "status": "success",
                "path": _pack(grid, path),
                "trace": _pack(grid, trace),
                "expanded": len(trace),
                "generated": len(seen),
                "length": len(path) - 1,
                "time_ms": (time.perf_counter() - t0) * 1000.0,
            }
        if cutoff and len(trace) >= cutoff:
            return {"status": "cutoff", "path": np.zeros((0, 2), int), "trace": _pack(grid, trace),
                    "expanded": len(trace), "generated": len(seen), "length": None,
                    "time_ms": (time.perf_counter() - t0) * 1000.0}
        for u in nbr[v]:
            u = int(u)
            if u in seen:
                continue
            seen.add(u)
            parent[u] = v
            if u not in rows_cache:
                rows_cache[u] = divmod(u, grid.width)
            counter += 1
            heapq.heappush(heap, (priority(*rows_cache[u]), counter, u))
    return {"status": "unreachable", "path": np.zeros((0, 2), int), "trace": _pack(grid, trace),
            "expanded": len(trace), "generated": len(seen), "length": None,
            "time_ms": (time.perf_counter() - t0) * 1000.0}


# --------------------------------------------------------------------------- #
# public searches
# --------------------------------------------------------------------------- #
def astar(grid: Grid, start, goal, cutoff: int = 0) -> dict:
    """A* with the Manhattan heuristic; this is the only optimality-claiming search."""
    t0 = time.perf_counter()
    s, t = grid.idx(*start), grid.idx(*goal)
    if not grid.free.flat[s] or not grid.free.flat[t]:
        return {"status": "invalid", "path": np.zeros((0, 2), int), "trace": np.zeros((0, 2), int),
                "expanded": 0, "generated": 0, "length": None, "time_ms": 0.0}
    gr, gc = goal
    g = {s: 0}
    parent = {s: -1}
    counter = 0
    heap = [(abs(start[0] - gr) + abs(start[1] - gc), 0, s)]
    closed = set()
    trace: list[int] = []
    nbr = grid._nbr
    while heap:
        _, _, v = heapq.heappop(heap)
        if v in closed:
            continue
        closed.add(v)
        trace.append(v)
        if v == t:
            path = []
            while v != -1:
                path.append(v)
                v = parent[v]
            path.reverse()
            return {"status": "success", "path": _pack(grid, path), "trace": _pack(grid, trace),
                    "expanded": len(trace), "generated": len(g), "length": int(g[t]),
                    "time_ms": (time.perf_counter() - t0) * 1000.0}
        if cutoff and len(trace) >= cutoff:
            break
        gv = g[v] + 1
        for u in nbr[v]:
            u = int(u)
            if u in closed or g.get(u, 1 << 30) <= gv:
                continue
            g[u] = gv
            parent[u] = v
            r, c = divmod(u, grid.width)
            counter += 1
            heapq.heappush(heap, (gv + abs(r - gr) + abs(c - gc), counter, u))
    status = "cutoff" if cutoff and len(trace) >= cutoff else "unreachable"
    return {"status": status, "path": np.zeros((0, 2), int), "trace": _pack(grid, trace),
            "expanded": len(trace), "generated": len(g), "length": None,
            "time_ms": (time.perf_counter() - t0) * 1000.0}


def guided_search(grid: Grid, start, goal, field: np.ndarray | None = None,
                  alpha: float = 0.0, gamma: float = 1.0, adaptive: bool = False,
                  alpha_far: float = 0.25, alpha_near: float = 0.75,
                  clip=None, weight: str = "linear", cutoff: int = 0) -> dict:
    """Best-first search on the guidance field, optionally fused with the goal distance."""
    d_hat, d_max = normalise_distance(grid, goal)
    if field is not None and not np.isfinite(field).any():
        raise ValueError("the guidance field has no finite cell; build it from a trajectory "
                         "that touches free space")
    f_hat = normalise_field(field, clip, weight) if field is not None else np.zeros((grid.height, grid.width))
    if adaptive:
        # far from the goal trust the field more, close to the goal trust the distance more
        alpha_map = alpha_far + (alpha_near - alpha_far) * (1.0 - d_hat)
    else:
        alpha_map = np.full_like(d_hat, float(alpha))
    pow_term = np.power(np.clip(d_hat, 0.0, 1.0), float(gamma))
    priority_map = alpha_map * pow_term - (1.0 - alpha_map) * f_hat
    return _best_first(grid, start, goal, lambda r, c: float(priority_map[r, c]), cutoff)
