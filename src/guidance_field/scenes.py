"""map families: complex occupancy environments instead of carved corridors.

A: random clutter of mixed-shape obstacles   (rect / rotated rect / ellipse / polygon /
                                              cross / L)
B: rooms separated by walls, joined by door openings, with furniture inside
C: multi-branch maze with widened passages, loops, dead ends and small rooms
D: structured topology: U / C / L / comb shapes, gate pairs, obstacle clusters, bottlenecks
"""

from __future__ import annotations

import numpy as np

from .grid import Grid
from .config import FREE_FRACTION_BANDS, MAP_TYPE_NAMES

FREE = 0
OCC = 1

SHAPE_KINDS = ("rect", "rotated_rect", "ellipse", "polygon", "cross", "ell")
STRUCTURE_KINDS = ("U", "C", "L", "comb", "gate", "cluster", "bottleneck")


# --------------------------------------------------------------------------- #
# rasterisation helpers (cell (r, c) is the unit square centred at (r, c))
# --------------------------------------------------------------------------- #
def _grid_coords(size: int):
    rr, cc = np.mgrid[0:size, 0:size]
    return rr.astype(np.float64), cc.astype(np.float64)


def fill_rect(occ: np.ndarray, r0: float, r1: float, c0: float, c1: float) -> None:
    h, w = occ.shape
    ra, rb = sorted((r0, r1))
    ca, cb = sorted((c0, c1))
    rs = slice(max(0, int(np.floor(ra + 0.5))), min(h, int(np.ceil(rb - 0.5)) + 1))
    cs = slice(max(0, int(np.floor(ca + 0.5))), min(w, int(np.ceil(cb - 0.5)) + 1))
    occ[rs, cs] = OCC


def fill_polygon(occ: np.ndarray, vertices) -> None:
    """Even-odd rasterisation of a simple polygon (vertices in ``(row, col)``)."""
    h, w = occ.shape
    rr, cc = _grid_coords(h)
    inside = np.zeros((h, w), dtype=bool)
    n = len(vertices)
    for i in range(n):
        y1, x1 = vertices[i]
        y2, x2 = vertices[(i + 1) % n]
        cond = (y1 > rr) != (y2 > rr)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x1 + (rr - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (cc < x_cross)
    occ[inside] = OCC


def fill_ellipse(occ: np.ndarray, center, radii, angle: float = 0.0) -> None:
    h, w = occ.shape
    rr, cc = _grid_coords(h)
    cr, cc0 = center
    rx, ry = radii
    ca, sa = np.cos(angle), np.sin(angle)
    dr, dc = rr - cr, cc - cc0
    u = dr * ca + dc * sa
    v = -dr * sa + dc * ca
    occ[(u / max(rx, 1e-6)) ** 2 + (v / max(ry, 1e-6)) ** 2 <= 1.0] = OCC


def _rotated_rect_polygon(center, half, angle: float):
    cr, cc = center
    hr, hc = half
    ca, sa = np.cos(angle), np.sin(angle)
    pts = []
    for sr, sc in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        dr, dc = sr * hr, sc * hc
        pts.append((cr + dr * ca - dc * sa, cc + dr * sa + dc * ca))
    return pts


def _free_components(occ: np.ndarray) -> int:
    return Grid(occ).components()[1]


def _keep_largest_free_component(occ: np.ndarray) -> int:
    """Fill every free blob except the largest one; returns how many cells were filled.

    Generators can leave unreachable slivers (a room corner sealed off by furniture, a
    pocket enclosed by two clutter shapes). They carry no trajectory information and
    would break the "single connected component" requirement, so they become wall.
    """
    labels, n = Grid(occ).components()
    if n <= 1:
        return 0
    counts = np.bincount(labels[labels >= 0])
    main = int(np.argmax(counts))
    stray = (labels >= 0) & (labels != main)
    removed = int(stray.sum())
    occ[stray] = OCC
    return removed


def _clearance_map(occ: np.ndarray) -> np.ndarray:
    """Geodesic distance from each free cell to the nearest obstacle (0 on obstacles)."""
    from collections import deque

    h, w = occ.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    dq = deque()
    for r in range(h):
        for c in range(w):
            if occ[r, c] == OCC:
                dist[r, c] = 0
                dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] < 0:
                dist[rr, cc] = dist[r, c] + 1
                dq.append((rr, cc))
    return dist


# --------------------------------------------------------------------------- #
# A -- random clutter of mixed shapes
# --------------------------------------------------------------------------- #
def generate_random_clutter_scene(rng: np.random.Generator, size: int) -> tuple[np.ndarray, dict]:
    occ = np.zeros((size, size), dtype=np.int8)
    target = float(rng.uniform(0.13, 0.24))
    shapes: list[dict] = []
    tries = 0
    while occ.sum() / occ.size < target and tries < 600:
        tries += 1
        kind = SHAPE_KINDS[int(rng.integers(len(SHAPE_KINDS)))]
        r0 = float(rng.uniform(2, size - 3))
        c0 = float(rng.uniform(2, size - 3))
        before = int(occ.sum())
        if kind == "rect":
            hh = float(rng.uniform(1.2, 5.5))
            ww = float(rng.uniform(1.2, 7.0))
            fill_rect(occ, r0 - hh, r0 + hh, c0 - ww, c0 + ww)
            params = {"half": [round(hh, 2), round(ww, 2)]}
        elif kind == "rotated_rect":
            hh = float(rng.uniform(1.0, 3.0))
            ww = float(rng.uniform(2.0, 7.0))
            ang = float(rng.uniform(0, np.pi))
            fill_polygon(occ, _rotated_rect_polygon((r0, c0), (hh, ww), ang))
            params = {"half": [round(hh, 2), round(ww, 2)], "angle": round(ang, 3)}
        elif kind == "ellipse":
            rx = float(rng.uniform(1.2, 4.0))
            ry = float(rng.uniform(1.2, 5.5))
            ang = float(rng.uniform(0, np.pi))
            fill_ellipse(occ, (r0, c0), (rx, ry), ang)
            params = {"radii": [round(rx, 2), round(ry, 2)], "angle": round(ang, 3)}
        elif kind == "polygon":
            n = int(rng.integers(3, 7))
            rad = float(rng.uniform(2.0, 4.5))
            base = float(rng.uniform(0, 2 * np.pi))
            verts = [
                (r0 + rad * np.sin(base + 2 * np.pi * k / n) * float(rng.uniform(0.7, 1.2)),
                 c0 + rad * np.cos(base + 2 * np.pi * k / n) * float(rng.uniform(0.7, 1.2)))
                for k in range(n)
            ]
            fill_polygon(occ, verts)
            params = {"n": n, "radius": round(rad, 2)}
        elif kind == "cross":
            t = float(rng.uniform(0.8, 1.8))
            arm = float(rng.uniform(2.0, 4.5))
            fill_rect(occ, r0 - arm, r0 + arm, c0 - t, c0 + t)
            fill_rect(occ, r0 - t, r0 + t, c0 - arm, c0 + arm)
            params = {"thickness": round(t, 2), "arm": round(arm, 2)}
        else:  # ell
            a = float(rng.uniform(2.0, 5.0))
            b = float(rng.uniform(2.0, 5.0))
            t = float(rng.uniform(0.8, 1.6))
            fill_rect(occ, r0 - a, r0 + a, c0 - t, c0 + t)
            fill_rect(occ, r0 + a - t, r0 + a + t, c0 - t, c0 + b)
            params = {"a": round(a, 2), "b": round(b, 2), "thickness": round(t, 2)}
        # reject shapes that changed nothing (off-map) or that swallow the border
        if int(occ.sum()) == before:
            continue
        if occ[0, :].sum() and occ[0, 0] and int(occ[0].sum()) == size and int(occ[:, 0].sum()) == size:
            pass
        shapes.append({"kind": kind, "center": [round(r0, 2), round(c0, 2)], **params})
    structure = {
        "shape_count": len(shapes),
        "shape_kinds": sorted({s["kind"] for s in shapes}),
        "shape_histogram": {k: sum(1 for s in shapes if s["kind"] == k) for k in SHAPE_KINDS},
    }
    return occ, structure


# --------------------------------------------------------------------------- #
# B -- rooms, walls, door openings, furniture
# --------------------------------------------------------------------------- #
def generate_room_corridor_scene(rng: np.random.Generator, size: int) -> tuple[np.ndarray, dict]:
    occ = np.ones((size, size), dtype=np.int8)  # walls first; rooms are cut out
    rooms: list[tuple[int, int, int, int]] = []  # r0, r1, c0, c1 (inclusive)

    def split(r0, r1, c0, c1, depth):
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if depth <= 0 or (h < 9 and w < 9) or (h < 14 and w < 14 and rng.random() < 0.35):
            if h >= 5 and w >= 5:
                rooms.append((r0, r1, c0, c1))
            return
        if h > w:
            cut = int(rng.integers(r0 + 4, r1 - 3))
            split(r0, cut - 1, c0, c1, depth - 1)
            split(cut + 1, r1, c0, c1, depth - 1)
        else:
            cut = int(rng.integers(c0 + 4, c1 - 3))
            split(r0, r1, c0, cut - 1, depth - 1)
            split(r0, r1, cut + 1, c1, depth - 1)

    split(1, size - 2, 1, size - 2, int(rng.integers(3, 5)))
    for r0, r1, c0, c1 in rooms:
        occ[r0 + 1 : r1, c0 + 1 : c1] = FREE  # leave a 1-cell wall inside the split line

    # door openings: connect room pairs that are separated by a single wall
    def wall_between(a, b):
        ar0, ar1, ac0, ac1 = a
        br0, br1, bc0, bc1 = b
        if ac1 + 2 == bc0 or bc1 + 2 == ac0:  # vertical wall between them
            lo, hi = max(ar0, br0), min(ar1, br1)
            if hi - lo >= 1:
                return "v", lo, hi
        if ar1 + 2 == br0 or br1 + 2 == ar0:  # horizontal wall
            lo, hi = max(ac0, bc0), min(ac1, bc1)
            if hi - lo >= 1:
                return "h", lo, hi
        return None

    doors = []
    connected_pairs = set()
    order = list(range(len(rooms)))
    rng.shuffle(order)
    for i in order:
        for j in order:
            if i >= j:
                continue
            spot = wall_between(rooms[i], rooms[j])
            if spot is None:
                continue
            axis, lo, hi = spot
            if rng.random() > 0.55:  # not every possible door is opened
                continue
            width = int(rng.integers(1, 4))
            span = hi - lo + 1
            width = max(1, min(width, span))
            pos = int(rng.integers(lo, lo + max(1, span - width + 1)))
            a = rooms[i]
            b = rooms[j]
            if axis == "v":
                col = a[3] + 1 if a[3] < b[2] else b[3] + 1
                occ[pos : pos + width, col] = FREE
            else:
                row = a[1] + 1 if a[1] < b[0] else b[1] + 1
                occ[row, pos : pos + width] = FREE
            doors.append({"rooms": [i, j], "axis": axis, "pos": pos, "width": width})
            connected_pairs.add((i, j))

    # guarantee connectivity: join still-separated room groups with a thin corridor
    grid = Grid(occ)
    for _ in range(24):
        labels, n = grid.components()
        free_labels = {int(labels[rooms[k][0] + 1, rooms[k][2] + 1]) for k in range(len(rooms))}
        if n == 1 or len(free_labels) <= 1:
            break
        la = sorted(free_labels)[0]
        target = next(l for l in sorted(free_labels) if l != la)
        # walk a straight corridor from a room in `la` to a room in `target`
        a_room = next(k for k in range(len(rooms)) if int(labels[rooms[k][0] + 1, rooms[k][2] + 1]) == la)
        b_room = next(k for k in range(len(rooms)) if int(labels[rooms[k][0] + 1, rooms[k][2] + 1]) == target)
        ra, ca = (rooms[a_room][0] + rooms[a_room][1]) // 2, (rooms[a_room][2] + rooms[a_room][3]) // 2
        rb, cb = (rooms[b_room][0] + rooms[b_room][1]) // 2, (rooms[b_room][2] + rooms[b_room][3]) // 2
        occ[min(ra, rb) : max(ra, rb) + 1, ca] = FREE
        occ[rb, min(ca, cb) : max(ca, cb) + 1] = FREE
        doors.append({"rooms": [a_room, b_room], "axis": "corridor", "pos": None, "width": 1})
        grid = Grid(occ)

    # furniture: small blocks inside rooms, kept away from the room walls
    furniture = 0
    for r0, r1, c0, c1 in rooms:
        for _ in range(int(rng.integers(0, 4))):
            hh = int(rng.integers(1, 3))
            ww = int(rng.integers(1, 3))
            if r1 - r0 < 6 or c1 - c0 < 6:
                continue
            rr = int(rng.integers(r0 + 2, r1 - hh - 1))
            cc = int(rng.integers(c0 + 2, c1 - ww - 1))
            occ[rr : rr + hh, cc : cc + ww] = OCC
            furniture += 1
    structure = {
        "room_count": len(rooms),
        "door_count": len(doors),
        "furniture_count": furniture,
        "rooms": [[int(v) for v in room] for room in rooms],
    }
    return occ, structure


# --------------------------------------------------------------------------- #
# C -- multi-branch maze with widened passages, loops, dead ends, small rooms
# --------------------------------------------------------------------------- #
def generate_complex_maze_scene(rng: np.random.Generator, size: int) -> tuple[np.ndarray, dict]:
    occ = np.ones((size, size), dtype=np.int8)
    n = (size - 2) // 2  # lattice cells at odd coordinates
    for i in range(n):
        for j in range(n):
            occ[2 * i + 1, 2 * j + 1] = FREE

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
        ni, nj = cand[int(rng.integers(len(cand)))]
        visited[ni, nj] = True
        occ[i + ni + 1, j + nj + 1] = FREE
        stack.append((ni, nj))

    loops = 0
    for i in range(n):
        for j in range(n):
            if i + 1 < n and rng.random() < 0.18:
                occ[2 * i + 2, 2 * j + 1] = FREE
                loops += 1
            if j + 1 < n and rng.random() < 0.18:
                occ[2 * i + 1, 2 * j + 2] = FREE
                loops += 1

    widened = 0
    for _ in range(int(rng.integers(6, 14))):
        i, j = int(rng.integers(1, n - 1)), int(rng.integers(1, n - 1))
        r, c = 2 * i + 1, 2 * j + 1
        if rng.random() < 0.5:
            fill_rect(occ, r - 1, r + 1, c - 1, c + 1)
        else:
            fill_rect(occ, r - 1, r + 1, c, c + 2)
        widened += 1

    rooms = 0
    for _ in range(int(rng.integers(3, 7))):
        i, j = int(rng.integers(1, n - 2)), int(rng.integers(1, n - 2))
        r, c = 2 * i + 1, 2 * j + 1
        fill_rect(occ, r - 1, r + 3, c - 1, c + 3)
        rooms += 1

    # dead ends: short pockets off a free cell
    dead_ends = 0
    for _ in range(int(rng.integers(8, 18))):
        i, j = int(rng.integers(1, n - 1)), int(rng.integers(1, n - 1))
        r, c = 2 * i + 1, 2 * j + 1
        if occ[r, c] != FREE:
            continue
        dr, dc = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(rng.integers(4))]
        length = int(rng.integers(2, 5))
        cells = []
        for k in range(length):
            rr, cc = r + dr * (k + 1), c + dc * (k + 1)
            if not (1 <= rr < size - 1 and 1 <= cc < size - 1) or occ[rr, cc] == FREE:
                break
            cells.append((rr, cc))
        if cells:
            for rr, cc in cells:
                occ[rr, cc] = FREE
            dead_ends += 1

    structure = {
        "loops": loops,
        "widened_cells": widened,
        "rooms": rooms,
        "dead_end_pockets": dead_ends,
        "lattice_cells": n * n,
    }
    return occ, structure


# --------------------------------------------------------------------------- #
# D -- structured topology: U / C / L / comb, gates, clusters, bottlenecks
# --------------------------------------------------------------------------- #
def _stamp_U(occ, r, c, h, w, t, ang_quarter):
    if ang_quarter in (0, 2):
        fill_rect(occ, r - h, r + h, c - w, c - w + t)
        fill_rect(occ, r - h, r + h, c + w - t, c + w)
        fill_rect(occ, r + h - t, r + h, c - w, c + w)
        if ang_quarter == 2:
            fill_rect(occ, r - h, r - h + t, c - w, c + w)
    else:
        fill_rect(occ, r - w, r - w + t, c - h, c + h)
        fill_rect(occ, r + w - t, r + w, c - h, c + h)
        fill_rect(occ, r - w, r + w, c + h - t, c + h)


def _stamp_L(occ, r, c, h, w, t, ang_quarter):
    fill_rect(occ, r - h, r + h, c - w, c - w + t)
    if ang_quarter in (0, 1):
        fill_rect(occ, r + h - t, r + h, c - w, c + w)
    else:
        fill_rect(occ, r - h, r - h + t, c - w, c + w)


def _stamp_C(occ, r, c, h, w, t, ang_quarter):
    _stamp_U(occ, r, c, h, w, t, ang_quarter)
    _stamp_L(occ, r, c, h, w, t, ang_quarter)


def generate_structured_obstacle_scene(rng: np.random.Generator, size: int) -> tuple[np.ndarray, dict]:
    occ = np.zeros((size, size), dtype=np.int8)
    counts = {k: 0 for k in STRUCTURE_KINDS}
    gates: list[dict] = []

    def trial(fn) -> bool:
        backup = occ.copy()
        fn()
        occ[0, :] = OCC
        occ[-1, :] = OCC
        occ[:, 0] = OCC
        occ[:, -1] = OCC
        if _free_components(occ) != 1:
            occ[:, :] = backup
            return False
        return True

    for _ in range(int(rng.integers(3, 6))):
        kind = STRUCTURE_KINDS[int(rng.integers(3))]
        r = float(rng.uniform(10, size - 11))
        c = float(rng.uniform(10, size - 11))
        h = float(rng.uniform(4, 9))
        w = float(rng.uniform(4, 9))
        t = float(rng.uniform(1.0, 2.0))
        q = int(rng.integers(4))
        fn = {"U": _stamp_U, "C": _stamp_C, "L": _stamp_L}[kind]
        if trial(lambda: fn(occ, r, c, h, w, t, q)):
            counts[kind] += 1

    for _ in range(int(rng.integers(2, 5))):  # comb / multi-rect cluster
        r = float(rng.uniform(8, size - 9))
        c = float(rng.uniform(8, size - 9))
        parts = int(rng.integers(3, 7))
        cells = []
        for _k in range(parts):
            dr = float(rng.uniform(-6, 6))
            dc = float(rng.uniform(-6, 6))
            hh = float(rng.uniform(1.0, 2.5))
            ww = float(rng.uniform(1.0, 3.5))
            cells.append((r + dr, c + dc, hh, ww))
        if trial(lambda: [fill_rect(occ, rr - hh, rr + hh, cc - ww, cc + ww) for rr, cc, hh, ww in cells]):
            counts["comb" if parts >= 4 else "cluster"] += 1

    for _ in range(int(rng.integers(2, 5))):  # gate pairs: two blocks with a narrow gap
        r = float(rng.uniform(8, size - 9))
        c = float(rng.uniform(8, size - 9))
        gap = float(rng.uniform(1.0, 2.0))
        length = float(rng.uniform(6, 14))
        horiz = bool(rng.integers(2))
        if trial(
            lambda: (
                fill_rect(occ, r - length, r + length, c - gap - 2, c - gap)
                if horiz
                else fill_rect(occ, r - gap - 2, r - gap, c - length, c + length),
                fill_rect(occ, r - length, r + length, c + gap, c + gap + 2)
                if horiz
                else fill_rect(occ, r + gap, r + gap + 2, c - length, c + length),
            )
        ):
            counts["gate"] += 1
            gates.append({"center": [round(r, 1), round(c, 1)], "gap": round(gap, 2), "horizontal": horiz})

    for _ in range(int(rng.integers(1, 4))):  # bottlenecks: two long walls with one narrow passage
        r = float(rng.uniform(12, size - 13))
        c = float(rng.uniform(12, size - 13))
        length = float(rng.uniform(10, 20))
        opening = float(rng.uniform(1.0, 2.0))
        off = float(rng.uniform(-length * 0.6, length * 0.6))

        def stamp_bottleneck():
            fill_rect(occ, r - length, r + length, c - 1, c + 1)
            fill_rect(occ, r - length, r + off - opening, c - 1, c + 1)
            fill_rect(occ, r + off + opening, r + length, c - 1, c + 1)

        if trial(stamp_bottleneck):
            counts["bottleneck"] += 1

    structure = {"structure_counts": counts, "structure_kinds": sorted(k for k, v in counts.items() if v), "gates": gates}
    return occ, structure


GENERATORS = {
    "A": generate_random_clutter_scene,
    "B": generate_room_corridor_scene,
    "C": generate_complex_maze_scene,
    "D": generate_structured_obstacle_scene,
}


# --------------------------------------------------------------------------- #
# start / goal selection + validation
# --------------------------------------------------------------------------- #
def _pick_endpoints(grid: Grid, occ: np.ndarray, map_type: str, rng: np.random.Generator, size: int):
    clearance = _clearance_map(occ)
    min_clear = 1 if map_type == "D" else 2
    free = np.argwhere(grid.free)
    ok = free[clearance[free[:, 0], free[:, 1]] >= min_clear]
    if len(ok) < 20:
        return None
    sample = ok[rng.choice(len(ok), size=min(48, len(ok)), replace=False)]
    idx = [grid.idx(int(r), int(c)) for r, c in sample]
    best_pair, best_score = None, -1.0
    for a in range(len(idx)):
        dist = grid.geodesic_distance([idx[a]])
        for b in range(len(idx)):
            if a == b:
                continue
            d_geo = int(dist[idx[b]])
            if d_geo < 0:
                continue
            d_eu = float(np.hypot(*(sample[a] - sample[b])))
            score = d_geo + 0.5 * d_eu  # far in both metrics
            if score > best_score:
                best_score = score
                best_pair = (tuple(int(v) for v in sample[a]), tuple(int(v) for v in sample[b]))
    return best_pair


def validate_scene(grid: Grid, occ: np.ndarray, start, goal, map_type: str, structure: dict, cfg) -> tuple[bool, dict, str]:
    h, w = occ.shape
    stats: dict = {}
    n_comp = grid.components()[1]
    free_fraction = float(grid.free.mean())
    stats["components"] = int(n_comp)
    stats["free_fraction"] = round(free_fraction, 4)
    stats["obstacle_cells"] = int((~grid.free).sum())
    if n_comp != 1:
        return False, stats, "free space is not a single connected component"
    lo, hi = FREE_FRACTION_BANDS[map_type]
    if not (lo <= free_fraction <= hi):
        return False, stats, f"free fraction {free_fraction:.3f} outside {lo}-{hi}"
    if stats["obstacle_cells"] < 0.06 * occ.size:
        return False, stats, "not enough obstacle structure"
    dist = grid.geodesic_distance([grid.idx(*start)])
    d_sg = int(dist[grid.idx(*goal)])
    stats["geodesic_sg"] = d_sg
    stats["euclid_sg"] = round(float(np.hypot(start[0] - goal[0], start[1] - goal[1])), 2)
    if d_sg < cfg.min_geodesic_sg:
        return False, stats, f"geodesic S-G {d_sg} < {cfg.min_geodesic_sg}"
    # "not one narrow corridor": enough open (>=3 free neighbours) free cells
    free = grid.free
    deg = np.zeros_like(free, dtype=np.int32)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        deg += np.roll(np.roll(free, dr, axis=0), dc, axis=1).astype(np.int32)
    open_ratio = float((deg[free] >= 3).mean())
    stats["open_cell_ratio"] = round(open_ratio, 4)
    if map_type != "C" and open_ratio < 0.25:
        return False, stats, f"only {open_ratio:.2f} open cells: map is nearly one corridor"
    stats["cycles"] = int(grid.cycle_rank())
    dead_mask, _depth = grid.leaf_prune()
    stats["dead_end_cells"] = int(dead_mask.sum())
    if map_type == "C" and (stats["cycles"] < 3 or stats["dead_end_cells"] < 10):
        return False, stats, "maze lacks loops or dead ends"
    if map_type == "A" and (structure["shape_count"] < 6 or len(structure["shape_kinds"]) < 3):
        return False, stats, "clutter map needs >=6 shapes of >=3 kinds"
    if map_type == "B" and (structure["room_count"] < 4 or structure["door_count"] < 3):
        return False, stats, "room map needs >=4 rooms and >=3 door openings"
    if map_type == "D":
        kinds = structure["structure_kinds"]
        if len(kinds) < 3 or structure["structure_counts"].get("gate", 0) < 1:
            return False, stats, "structured map needs >=3 structure kinds and >=1 gate"
    return True, stats, ""


def generate_scene(scene_index: int, map_type: str, seed: int, cfg) -> dict:
    """Generate one validated map; records ``attempt`` and raises if it never validates."""
    size = cfg.grid_size
    rng = np.random.default_rng(seed)
    reason = ""
    for attempt in range(cfg.max_scene_attempts):
        occ, structure = GENERATORS[map_type](rng, size)
        occ[0, :] = OCC
        occ[-1, :] = OCC
        occ[:, 0] = OCC
        occ[:, -1] = OCC
        removed = _keep_largest_free_component(occ)
        structure["isolated_free_cells_removed"] = removed
        grid = Grid(occ)
        picked = _pick_endpoints(grid, occ, map_type, rng, size)
        if picked is None:
            reason = "no candidate endpoints"
            continue
        start, goal = picked
        if start == goal:
            continue
        ok, stats, reason = validate_scene(grid, occ, start, goal, map_type, structure, cfg)
        if not ok:
            continue
        meta = {
            "scene_id": f"scene_{scene_index:03d}",
            "index": scene_index,
            "type": map_type,
            "type_name": MAP_TYPE_NAMES[map_type],
            "seed": int(seed),
            "attempt": attempt,
            "size": size,
            "start": [start[0], start[1]],
            "goal": [goal[0], goal[1]],
            "stats": stats,
            "structure": structure,
        }
        return {"occupancy": occ.astype(np.int8), "metadata": meta}
    raise RuntimeError(f"map family {map_type} failed to generate a valid instance (seed={seed}): {reason}")
