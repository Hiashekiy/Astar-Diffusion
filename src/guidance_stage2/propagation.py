"""Alternative propagators for the max-plus guidance field.

Three interchangeable implementations of

    F(x) = max_i [ w_i - lam * d_free(x, s_i) ]

* ``propagate_heap``       -- multi-source max-Dijkstra (exact baseline)
* ``propagate_vectorized`` -- synchronous max-plus relaxation over the whole grid
* ``propagate_bucket``     -- Dial-style bucket queue on the equivalent shortest-path form
                              ``C(x) = min_i[d_free(x, s_i) + (1 - w_i)/lam]``, ``F = 1 - lam C``

All three return NaN on obstacles *and* on free cells no source can reach: unreachable cells
simply stay at ``-inf`` and are masked at the end, so no connected-component pre-pass is
needed.  ``propagate_vectorized`` cannot report the dominant source cheaply and refuses
``return_owner``.
"""

from __future__ import annotations

import heapq

import numpy as np

from guidance_stage1.grid import Grid

NEG_INF = -np.inf


def _seed(grid: Grid, samples, weights) -> tuple[np.ndarray, np.ndarray]:
    field = np.full((grid.height, grid.width), NEG_INF, dtype=np.float64)
    owner = np.full((grid.height, grid.width), -1, dtype=np.int32)
    samples = np.asarray(samples, dtype=int).reshape(-1, 2)
    for i, (r, c) in enumerate(samples):
        r, c = int(r), int(c)
        w = float(weights[i])
        if w > field[r, c]:
            field[r, c] = w
            owner[r, c] = i
    return field, owner


def _finalise(field: np.ndarray, owner, free: np.ndarray, return_owner: bool):
    out = np.where(free & (field > NEG_INF), field, np.nan)
    if not return_owner:
        return out
    own = np.where(free & (owner >= 0), owner, -1) if owner is not None else np.full(free.shape, -1, np.int32)
    return out, own


def propagate_heap(grid: Grid, samples, weights, lam: float, return_owner: bool = False):
    """Multi-source max-Dijkstra on a max-heap (exact baseline)."""
    samples = np.asarray(samples, dtype=int).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    field2d, owner2d = _seed(grid, samples, weights)
    field = field2d.ravel().copy()
    owner = owner2d.ravel().copy()
    heap = []
    queued = set()
    for r, c in samples:
        idx = grid.idx(int(r), int(c))
        if idx not in queued and field[idx] > NEG_INF:
            queued.add(idx)
            heap.append((-field[idx], idx))
    heapq.heapify(heap)
    nbr = grid._nbr  # precomputed neighbour table: no numpy call per node
    while heap:
        neg_v, v = heapq.heappop(heap)
        v_val = -neg_v
        if v_val < field[v]:
            continue
        new_value = v_val - lam
        for u in nbr[v]:
            if u < 0:
                continue
            if new_value > field[u]:
                field[u] = new_value
                owner[u] = owner[v]
                heapq.heappush(heap, (-new_value, u))
    return _finalise(field.reshape(field2d.shape), owner.reshape(field2d.shape), grid.free, return_owner)


def propagate_vectorized(grid: Grid, samples, weights, lam: float, return_owner: bool = False,
                         steps_per_round: int = 4, max_rounds: int = 200000) -> np.ndarray:
    """``F <- max(F, shift(F) - lam)`` over the whole grid until nothing changes.

    Every step is a fully synchronous sweep (each direction reads the *previous* field), so
    the result does not depend on the update order.  ``steps_per_round`` applies the operator
    several times before the convergence test: that changes only how often we test, not the
    fixed point, and removes most of the per-round Python overhead.

    Cost is ``O(eccentricity * H * W)`` versus the heap's ``O(V log V)``: this wins where the
    geodesic diameter is small relative to the map area (open clutter, rooms) and loses on
    long winding corridors -- see ``scripts/benchmark_propagation.py``.
    """
    if return_owner:
        raise ValueError("propagate_vectorized cannot report the dominant source; use the heap variant")
    samples = np.asarray(samples, dtype=int).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    field, _ = _seed(grid, samples, weights)
    free = grid.free
    field[~free] = NEG_INF
    work = np.empty_like(field)
    for _round in range(max_rounds):
        changed = False
        for _step in range(max(1, steps_per_round)):
            np.copyto(work, field)
            np.maximum(work[1:, :], field[:-1, :] - lam, out=work[1:, :])
            np.maximum(work[:-1, :], field[1:, :] - lam, out=work[:-1, :])
            np.maximum(work[:, 1:], field[:, :-1] - lam, out=work[:, 1:])
            np.maximum(work[:, :-1], field[:, 1:] - lam, out=work[:, :-1])
            work[~free] = NEG_INF
            if np.array_equal(work, field):
                field, work = work, field
                break
            field, work = work, field
            changed = True
        if not changed:
            break
    return _finalise(field, None, free, False)


def propagate_bucket(grid: Grid, samples, weights, lam: float, return_owner: bool = False,
                     scale: int = 16):
    """``C(x) = min_i[d + (1-w_i)/lam]`` via integer buckets, ``F = 1 - lam*C``.

    Keys are ``scale*d + round(scale*(1-w)/lam)``, so the quantisation error of F is at most
    ``lam/(2*scale)``.  A single ring of ``scale+1`` buckets is not usable here: the initial
    keys (source offsets) are arbitrarily spaced, so a ring would recycle a slot before its
    entry is due.  The queue therefore keeps a min-heap of the *levels* that currently hold
    nodes, plus one list per level.
    """
    samples = np.asarray(samples, dtype=int).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    free_flat = grid.free.ravel()
    big = np.iinfo(np.int64).max
    size = grid.size
    key = np.full(size, big, dtype=np.int64)
    owner_flat = np.full(size, -1, dtype=np.int32)

    levels: dict[int, list[int]] = {}
    level_heap: list[int] = []

    def push(level: int, node: int) -> None:
        bucket = levels.get(level)
        if bucket is None:
            levels[level] = [node]
            heapq.heappush(level_heap, level)
        else:
            bucket.append(node)

    for i, (r, c) in enumerate(samples):
        idx = grid.idx(int(r), int(c))
        offset = int(round(scale * (1.0 - float(weights[i])) / lam))
        if offset < key[idx]:
            key[idx] = offset
            owner_flat[idx] = i
            push(offset, idx)

    nbr = grid._nbr
    bits = np.zeros(size, dtype=bool)
    while level_heap:
        level = heapq.heappop(level_heap)
        bucket = levels.pop(level)
        next_level = level + scale
        for v in bucket:
            if bits[v] or key[v] != level:
                continue
            bits[v] = True
            for u in nbr[v]:
                if u < 0:
                    continue
                if not bits[u] and next_level < key[u]:
                    key[u] = next_level
                    owner_flat[u] = owner_flat[v]
                    push(next_level, u)

    reached = bits & free_flat
    field = np.where(reached, 1.0 - lam * (key.astype(np.float64) / scale), NEG_INF)
    owner = np.where(reached, owner_flat, -1)
    return _finalise(field.reshape(grid.free.shape), owner.reshape(grid.free.shape), grid.free, return_owner)


def propagate_auto(grid: Grid, samples, weights, lam: float, return_owner: bool = False, **kwargs):
    """Pick the exact implementation that wins for this map shape.

    Measured on the 20 stage-2A maps: the synchronous method is 2.6-2.9x faster on the open
    and room-like families (free fraction >= 0.55) but marginally slower on the dense mazes,
    because its cost scales with the geodesic diameter x area while Dijkstra scales with the
    number of free cells.
    """
    if return_owner:
        return propagate_heap(grid, samples, weights, lam, return_owner=True)
    if float(grid.free.mean()) >= 0.55:
        return propagate_vectorized(grid, samples, weights, lam)
    return propagate_heap(grid, samples, weights, lam)


PROPAGATORS = {
    "heap": propagate_heap,
    "vectorized": propagate_vectorized,
    "bucket": propagate_bucket,
    "auto": propagate_auto,
}


def propagate(grid: Grid, samples, weights, lam: float, method: str = "vectorized",
              return_owner: bool = False, **kwargs):
    """Dispatch on ``method``; the default is the fastest *exact* implementation."""
    if method not in PROPAGATORS:
        raise KeyError(f"unknown propagation method {method!r}")
    return PROPAGATORS[method](grid, samples, weights, lam, return_owner=return_owner, **kwargs)
