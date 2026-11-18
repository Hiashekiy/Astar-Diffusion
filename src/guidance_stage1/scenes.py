"""Four non-perfect maze families (A: dual corridor, B: detour block, C: dead ends,
D: multi-branch lattice) plus structural validation of every generated instance."""

from __future__ import annotations

import numpy as np

from .config import FREE, OCCUPIED
from .grid import Grid, detour_profile

SCENE_TYPE_NAMES = {
    "A": "dual_corridor",
    "B": "detour_block",
    "C": "dead_end_branches",
    "D": "multi_branch_maze",
}


# --------------------------------------------------------------------------- #
# carving helpers (every generator starts from an all-wall grid)
# --------------------------------------------------------------------------- #
def _carve_h(occ: np.ndarray, row: int, c0: int, c1: int, t: int = 1) -> None:
    occ[row : row + t, min(c0, c1) : max(c0, c1) + 1] = FREE


def _carve_v(occ: np.ndarray, col: int, r0: int, r1: int, t: int = 1) -> None:
    occ[min(r0, r1) : max(r0, r1) + 1, col : col + t] = FREE


def _carve_rect(occ: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    occ[min(r0, r1) : max(r0, r1) + 1, min(c0, c1) : max(c0, c1) + 1] = FREE


def _free_cells(occ: np.ndarray) -> list[tuple[int, int]]:
    rs, cs = np.nonzero(occ == FREE)
    return list(zip(rs.tolist(), cs.tolist()))


def _stub_is_clean(cells: list[tuple[int, int]], free_before: np.ndarray) -> bool:
    """True when everything past the entry cell touches no pre-existing free space."""
    h, w = free_before.shape
    for r, c in cells[1:]:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                return False
            if free_before[rr, cc]:
                return False
    return True


def _add_stub(
    occ: np.ndarray,
    rng: np.random.Generator,
    spine: list[tuple[int, int]],
    length_range: tuple[int, int] = (6, 13),
    max_tries: int = 60,
) -> list[tuple[int, int]] | None:
    """Hang one dead-end pocket off the existing free space.

    The pocket is guaranteed to terminate (never reconnects), so it contributes a
    genuine dead end that a greedy planner can be lured into.
    """
    free_before = occ == FREE
    h, w = occ.shape
    for _ in range(max_tries):
        r0, c0 = spine[int(rng.integers(len(spine)))]
        if not free_before[r0, c0]:
            continue
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        rng.shuffle(dirs)
        for dr, dc in dirs[:2]:
            length = int(rng.integers(*length_range))
            cells: list[tuple[int, int]] = []
            rr, cc = r0, c0
            dr_, dc_ = dr, dc
            ok = True
            for k in range(length):
                if k == length // 2 and rng.random() < 0.5:
                    dr_, dc_ = dc_, dr  # single 90-degree elbow
                rr, cc = rr + dr_, cc + dc_
                if not (1 <= rr < h - 1 and 1 <= cc < w - 1) or occ[rr, cc] == FREE:
                    ok = False
                    break
                cells.append((rr, cc))
            if not ok or len(cells) < length:
                continue
            if _stub_is_clean(cells, free_before):
                for rr, cc in cells:
                    occ[rr, cc] = FREE
                return cells
    return None


# --------------------------------------------------------------------------- #
# scene A -- two parallel corridors of clearly different length
# --------------------------------------------------------------------------- #
def _scene_a(rng: np.random.Generator, size: int) -> np.ndarray:
    occ = np.full((size, size), OCCUPIED, dtype=np.int8)
    lo, hi = 2, size - 3

    ra = int(rng.integers(9, 22))  # upper corridor (direct)
    rb = int(rng.integers(30, 40))  # lower corridor (has a vertical step)
    h1 = int(rng.integers(1, 3))
    h2 = int(rng.integers(1, 3))
    dy = int(rng.integers(9, 16))  # lower corridor drops by dy half way
    c_mid = int(rng.integers(26, 38))

    g_top = lo + 7   # gap between the chamber and the corridor entries
    g_bot = hi - 7

    _carve_h(occ, ra, g_top, g_bot, h1)                       # straight upper corridor
    _carve_h(occ, rb, g_top, c_mid, h2)                       # lower corridor, first leg
    _carve_v(occ, c_mid, rb, rb + dy + h2 - 1, h2)            # lower corridor, step down
    _carve_h(occ, rb + dy, c_mid, g_bot, h2)                  # lower corridor, second leg

    bottom_row = rb + dy + h2 - 1
    for c0, c1 in ((lo, g_top), (g_bot, hi)):                 # S / G chambers
        _carve_rect(occ, ra - 2, bottom_row + 2, c0, c1)

    spine = _free_cells(occ)
    for _ in range(int(rng.integers(2, 5))):
        _add_stub(occ, rng, spine, (4, 9))
        spine = _free_cells(occ)

    # S and G sit level, between the two corridor mouths: the upper (direct) and the
    # lower (stepped) channel then genuinely compete, differing by exactly 2*dy.
    r = int(np.clip(round(0.5 * (ra + rb + h2 - 1)) + int(rng.integers(-2, 3)), ra - 1, bottom_row + 1))
    start = (r, int(rng.integers(lo + 1, g_top)))
    goal = (r, int(rng.integers(g_bot, hi)))
    return occ, start, goal


# --------------------------------------------------------------------------- #
# scene B -- one large central block, top and bottom detours of unequal cost
# --------------------------------------------------------------------------- #
def _scene_b(rng: np.random.Generator, size: int) -> np.ndarray:
    occ = np.full((size, size), OCCUPIED, dtype=np.int8)
    lo, hi = 2, size - 3
    rt = int(rng.integers(6, 13))
    rb = int(rng.integers(50, 57))
    h = int(rng.integers(1, 3))

    _carve_h(occ, rt, lo, hi, h)  # top bypass corridor
    _carve_h(occ, rb, lo, hi, h)  # bottom bypass corridor
    _carve_v(occ, lo, rt, rb + h - 1, 4)   # left shaft joins both
    _carve_v(occ, hi - 3, rt, rb + h - 1, 4)  # right shaft joins both

    # bump in the bottom corridor -> the lower detour is strictly longer
    du = int(rng.integers(16, 27))
    c1 = int(rng.integers(20, 32))
    c2 = c1 + int(rng.integers(8, 16))
    _carve_v(occ, c1, rb - du, rb + h - 1, h)
    _carve_h(occ, rb - du, c1, c2 + h - 1, h)
    _carve_v(occ, c2, rb - du, rb + h - 1, h)

    spine = _free_cells(occ)
    for _ in range(int(rng.integers(2, 4))):
        _add_stub(occ, rng, spine, (4, 9))
        spine = _free_cells(occ)

    # Level S / G: the top and the bottom bypass start equidistant, so their length
    # difference is the forced bump (2*du) and nothing else.
    r = int(np.clip(round(0.5 * (rt + rb + h - 1)) + int(rng.integers(-2, 3)), rt, rb + h - 1))
    start = (r, int(rng.integers(lo + 1, lo + 3)))
    goal = (r, int(rng.integers(hi - 2, hi)))
    return occ, start, goal


# --------------------------------------------------------------------------- #
# scene C -- crooked main corridor, tempting dead ends, one bypass loop
# --------------------------------------------------------------------------- #
def _scene_c(rng: np.random.Generator, size: int) -> np.ndarray:
    occ = np.full((size, size), OCCUPIED, dtype=np.int8)
    lo, hi = 2, size - 3
    t = int(rng.integers(2, 4))

    row = int(rng.integers(20, 44))
    pts = [(row, lo)]
    col = lo
    while col < hi:
        col = min(hi, col + int(rng.integers(5, 11)))
        pts.append((row, col))
        if col < hi and rng.random() < 0.7:
            row = int(np.clip(row + rng.choice([-1, 1]) * rng.integers(5, 12), 5, size - 6))
            pts.append((row, col))
    for a, b in zip(pts, pts[1:]):
        if a[0] == b[0]:
            _carve_h(occ, a[0], a[1], b[1], t)
        else:
            _carve_v(occ, a[1], a[0], b[0], t)

    main_cells = _free_cells(occ)
    for _ in range(int(rng.integers(4, 7))):
        _add_stub(occ, rng, main_cells, (7, 16))
    # bypasses keep the map non-perfect: pockets that reconnect much further along
    for _ in range(int(rng.integers(1, 3))):
        for _ in range(8):
            if _add_bypass(occ, rng, main_cells, (14, 26)):
                break
    # S / G at the two ends of the main corridor
    start = min(main_cells, key=lambda cell: cell[1])
    goal = max(main_cells, key=lambda cell: cell[1])
    return occ, start, goal


def _add_bypass(
    occ: np.ndarray,
    rng: np.random.Generator,
    spine: list[tuple[int, int]],
    length_range: tuple[int, int],
) -> bool:
    """Carve a detour that leaves the corridor and rejoins it much further along."""
    free_before = occ == FREE
    h, w = occ.shape
    for _ in range(40):
        r0, c0 = spine[int(rng.integers(len(spine)))]
        if not free_before[r0, c0]:
            continue
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        rng.shuffle(dirs)
        dr, dc = dirs[0]
        length = int(rng.integers(*length_range))
        rr, cc = r0, c0
        cells: list[tuple[int, int]] = []
        dr_, dc_ = dr, dc
        for k in range(length):
            if k in (length // 3, 2 * length // 3):
                dr_, dc_ = dc_, dr
            rr, cc = rr + dr_, cc + dc_
            if not (1 <= rr < h - 1 and 1 <= cc < w - 1) or occ[rr, cc] == FREE:
                break
            cells.append((rr, cc))
        if len(cells) < length_range[0]:
            continue
        # the last cell must touch the corridor -> closes a loop
        tip = cells[-1]
        touches = [
            (tip[0] + dr, tip[1] + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if free_before[tip[0] + dr, tip[1] + dc]
        ]
        if not touches or not _stub_is_clean(cells[:-1], free_before):
            continue
        for rr, cc in cells:
            occ[rr, cc] = FREE
        return True
    return False


# --------------------------------------------------------------------------- #
# scene D -- lattice maze with many loops, rooms, blocks and dead ends
# --------------------------------------------------------------------------- #
def _scene_d(rng: np.random.Generator, size: int) -> np.ndarray:
    occ = np.full((size, size), OCCUPIED, dtype=np.int8)
    n = (size - 2) // 2  # lattice cells at odd coordinates 1, 3, ..., 2n-1
    for i in range(n):
        for j in range(n):
            occ[2 * i + 1, 2 * j + 1] = FREE

    # randomised DFS spanning tree
    visited = np.zeros((n, n), dtype=bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        i, j = stack[-1]
        cand = [
            (i + di, j + dj)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= i + di < n and 0 <= j + dj < n and not visited[i + di, j + dj]
        ]
        if not cand:
            stack.pop()
            continue
        k = int(rng.integers(len(cand)))
        ni, nj = cand[k]
        visited[ni, nj] = True
        occ[i + ni + 1, j + nj + 1] = FREE  # knock the wall between cells
        stack.append((ni, nj))

    # loops: reopen a fraction of the remaining walls -> several routes, not a tree
    p_loop = float(rng.uniform(0.10, 0.20))
    for i in range(n):
        for j in range(n):
            if i + 1 < n and rng.random() < p_loop:
                occ[2 * i + 2, 2 * j + 1] = FREE
            if j + 1 < n and rng.random() < p_loop:
                occ[2 * i + 1, 2 * j + 2] = FREE

    # small rooms widen some junctions
    for _ in range(int(rng.integers(3, 6))):
        i, j = int(rng.integers(1, n - 2)), int(rng.integers(1, n - 2))
        _carve_rect(occ, 2 * i, 2 * i + 2, 2 * j, 2 * j + 2)

    # isolated blocks (kept only while free space stays connected)
    for _ in range(int(rng.integers(6, 14))):
        i, j = int(rng.integers(1, n - 1)), int(rng.integers(1, n - 1))
        r, c = 2 * i + 1, 2 * j + 1
        if occ[r, c] != FREE:
            continue
        occ[r, c] = OCCUPIED
        if Grid(occ).components()[1] != 1:
            occ[r, c] = FREE

    spine = [cell for cell in _free_cells(occ) if occ[cell[0], cell[1]] == FREE]
    for _ in range(int(rng.integers(1, 4))):
        _add_stub(occ, rng, spine, (5, 11))
        spine = _free_cells(occ)
    return occ, None, None  # endpoints chosen later by farthest-pair search


_GENERATORS = {"A": _scene_a, "B": _scene_b, "C": _scene_c, "D": _scene_d}


# --------------------------------------------------------------------------- #
# start / goal placement + validation
# --------------------------------------------------------------------------- #
def _farthest_pair(grid: Grid, rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
    free = _free_cells(grid.occupancy)
    sample = [free[int(i)] for i in rng.choice(len(free), size=min(40, len(free)), replace=False)]
    best = (sample[0], sample[0])
    best_d = -1
    for cell in sample:
        dist = grid.geodesic_distance([grid.idx(*cell)])
        for other in sample:
            d = int(dist[grid.idx(*other)])
            if d > best_d:
                best_d, best = d, (cell, other)
    return best


def generate_scene(scene_index: int, scene_type: str, seed: int, cfg) -> dict:
    """Generate one validated instance; retries with fresh randomness until valid."""
    size = cfg.grid_size
    rng = np.random.default_rng(seed)
    for attempt in range(60):
        occ, start, goal = _GENERATORS[scene_type](rng, size)
        occ[0, :] = occ[-1, :] = OCCUPIED
        occ[:, 0] = occ[:, -1] = OCCUPIED
        grid = Grid(occ)
        labels, n_comp = grid.components()
        free_frac = float(grid.free_fraction())
        if n_comp != 1 or not (0.03 <= free_frac <= 0.80):
            continue
        if start is None:
            start, goal = _farthest_pair(grid, rng)
        if not (grid.is_free(*start) and grid.is_free(*goal)) or start == goal:
            continue
        profile = detour_profile(grid, start, goal)
        dist_sg = profile["d_sg"]
        if not np.isfinite(dist_sg) or dist_sg < cfg.min_geodesic_sg:
            continue
        detour = profile["detour"]
        detour = np.where(np.isfinite(detour), detour, 0.0)
        pockets = int((detour >= 10.0).sum())
        max_detour = float(detour.max())
        # a Poor-grade detour has to be reachable: rho_max >= 1 + max_detour / L*
        if scene_type in ("C", "D") and pockets < 25:
            continue
        if 1.0 + max_detour / max(dist_sg, 1.0) < 1.35:
            continue
        _, peel_depth = grid.leaf_prune()
        record = {
            "scene_id": f"scene_{scene_index:03d}",
            "index": scene_index,
            "type": scene_type,
            "type_name": SCENE_TYPE_NAMES[scene_type],
            "seed": int(seed),
            "attempt": attempt,
            "size": size,
            "start": [int(start[0]), int(start[1])],
            "goal": [int(goal[0]), int(goal[1])],
            "stats": {
                "free_fraction": round(free_frac, 4),
                "components": int(n_comp),
                "cycles": int(grid.cycle_rank()),
                "pocket_cells_detour10": pockets,
                "max_detour": round(max_detour, 1),
                "max_detour_ratio": round(1.0 + max_detour / max(dist_sg, 1.0), 3),
                "geodesic_sg": int(dist_sg),
                "peel_depth_max": int(peel_depth.max()),
            },
        }
        return {"occupancy": occ.astype(np.int8), "metadata": record}
    raise RuntimeError(f"could not generate a valid {scene_type} scene (seed={seed})")
