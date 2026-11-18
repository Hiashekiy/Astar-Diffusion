"""Stage-2A figures: maps, colliding trajectories, the three source methods, and fields.

Layout follows the plan: every comparison panel keeps the same map, the same raw
trajectory and the same lambda; the raw wall-crossing geometry is always drawn on top
(collision segments in red) so the reader never sees a repaired trajectory pretending to
be the input.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from guidance_stage2.collision import arc_lengths, point_at_arc_length

BG = "white"
FIELD_CMAP = "viridis"
OWNER_CMAP = "plasma"
FREE_LIGHT = 0.93
OBS_DARK = 0.10
COL_FREE = "#1f77b4"
COL_COLLIDE = "#d62728"
COL_BRIDGE = "#2ca02c"
COL_PROJECT = "#ff7f0e"
GRADE_COLORS = {
    "feasible": "#1f77b4",
    "mild_collision": "#2ca02c",
    "moderate_collision": "#ff7f0e",
    "severe_collision": "#d62728",
}


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def _bg(ax, occ):
    ax.imshow(np.where(occ == 1, OBS_DARK, FREE_LIGHT), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")


def _colored_segments(raw: np.ndarray, intervals):
    """Split the polyline exactly at collision boundaries and colour each piece.

    A midpoint test is not enough: a long shortcut segment can cross several walls while
    its own midpoint stays in free space.  Cutting at every ``s_in`` / ``s_out`` makes the
    red/blue decision exact per sub-segment.
    """
    cum = arc_lengths(raw)
    segs, colors = [], []
    for j in range(len(raw) - 1):
        s0, s1 = float(cum[j]), float(cum[j + 1])
        if s1 - s0 <= 1e-12:
            continue
        cuts = {s0, s1}
        for iv in intervals:
            for bound in (iv["s_in"], iv["s_out"]):
                if s0 + 1e-12 < bound < s1 - 1e-12:
                    cuts.add(float(bound))
        pts = sorted(cuts)
        for a, b in zip(pts, pts[1:]):
            mid = 0.5 * (a + b)
            inside = any(iv["s_in"] - 1e-9 <= mid <= iv["s_out"] + 1e-9 for iv in intervals)
            segs.append(
                np.array([point_at_arc_length(raw, cum, a), point_at_arc_length(raw, cum, b)])
            )
            colors.append(COL_COLLIDE if inside else COL_FREE)
    return segs, colors


def _collide_flags(raw: np.ndarray, intervals) -> np.ndarray:
    """Per original segment: does it overlap any collision interval at all?"""
    cum = arc_lengths(raw)
    flags = np.zeros(max(len(raw) - 1, 0), dtype=bool)
    for j in range(len(raw) - 1):
        s0, s1 = float(cum[j]), float(cum[j + 1])
        flags[j] = any(iv["s_in"] < s1 - 1e-9 and iv["s_out"] > s0 + 1e-9 for iv in intervals)
    return flags


def draw_trajectory(ax, raw: np.ndarray, intervals, lw: float = 1.8, alpha: float = 1.0,
                    halo: bool = False):
    """Continuous polyline with free parts in blue and collision parts in red."""
    if len(raw) < 2:
        return
    segs, colors = _colored_segments(raw, intervals)
    if not segs:
        return
    if halo:
        ax.add_collection(LineCollection(segs, colors="black", linewidths=lw + 1.4, alpha=0.45, zorder=4))
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw, alpha=alpha, capstyle="round", zorder=5))


def _mark(ax, rc, color, marker="o", ms=7, label=None):
    ax.plot(rc[1], rc[0], marker, mfc=color, mec="white", mew=1.4, ms=ms, zorder=8, label=label)


def draw_traversed_cells(ax, cells, color=COL_COLLIDE, alpha=0.55):
    """Translucent patches on the obstacle cells the trajectory really passes through.

    Without this the red segment sits on a one-cell-wide dark seam and is hard to read as
    "inside an obstacle"; the patches make the overlap unmistakable.
    """
    from matplotlib.patches import Rectangle

    for r, c, _depth in cells:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, facecolor=color, edgecolor="white",
                               lw=0.6, alpha=alpha, zorder=6))


def _label_point(ax, rc, text, color, offset=(9, 9)):
    """Bold letter next to a marker, stroked so it survives any background colour."""
    import matplotlib.patheffects as pe

    ax.annotate(
        text, (rc[1], rc[0]), xytext=offset, textcoords="offset points",
        fontsize=15, fontweight="bold", color="white", zorder=9, ha="left", va="bottom",
        path_effects=[pe.withStroke(linewidth=3.2, foreground="black")],
    )


def _field_panel(ax, occ, field, title, kind, raw=None, intervals=None, support=None,
                 vmin=None, vmax=None, sources=None, start=None, goal=None, label_endpoints=True):
    _bg(ax, occ)
    if vmin is None or vmax is None:
        vmin, vmax = display_range(kind)
    data = np.where(np.isfinite(field), np.clip(field, vmin, vmax), np.nan)
    image = ax.imshow(data, cmap=FIELD_CMAP, vmin=vmin, vmax=vmax, interpolation="nearest", alpha=0.9)
    if sources is not None and len(sources):
        src = np.asarray(sources).reshape(-1, 2)
        ax.scatter(src[:, 1], src[:, 0], s=5, c="white", edgecolors="black", linewidths=0.25, zorder=6)
    if support is not None and len(support) > 1:
        ax.plot(support[:, 1], support[:, 0], "-", color=COL_BRIDGE, lw=1.0, alpha=0.9, zorder=5)
    if raw is not None:
        draw_trajectory(ax, raw, intervals or [], lw=1.2, alpha=0.95)
    if start is not None:
        _mark(ax, start, "#2b6cff", ms=11)
        if label_endpoints:
            _label_point(ax, start, "S", "#2b6cff", offset=(5, 4))
    if goal is not None:
        _mark(ax, goal, "#ff3b30", ms=11)
        if label_endpoints:
            _label_point(ax, goal, "G", "#ff3b30", offset=(5, 4))
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def display_range(kind: str) -> tuple[float, float]:
    """Colour range that shows the mechanism, with the far-field tail clipped."""
    return (0.50, 1.0) if kind == "uniform" else (-0.10, 1.0)


# --------------------------------------------------------------------------- #
# §8.1 map gallery
# --------------------------------------------------------------------------- #
def figure_map_gallery(out_path, scenes: list[dict], cfg) -> None:
    cols = cfg.instances_per_type
    rows = max(1, len(scenes) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.9 * rows), dpi=110)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, sc in zip(axes.ravel(), scenes):
        ax.axis("on")
        occ, md = sc["occupancy"], sc["metadata"]
        _bg(ax, occ)
        _mark(ax, md["start"], "#2b6cff", ms=6)
        _mark(ax, md["goal"], "#ff3b30", ms=6)
        st = md["stats"]
        ax.set_title(
            f"{md['scene_id']}  {md['type']} {md['type_name']}\n"
            f"free={st['free_fraction']:.2f}  d(S,G)={st['geodesic_sg']}  seed={md['seed']}",
            fontsize=7.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Stage 2A map gallery: 4 families x 5 instances (S blue, G red)", fontsize=12, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# §8.2 graded trajectories of one map
# --------------------------------------------------------------------------- #
def figure_trajectories(out_path, occ, md, trajectories: dict, grade_order) -> None:
    fig, axes = plt.subplots(1, len(grade_order), figsize=(4.5 * len(grade_order), 4.9), dpi=110)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes)
    for ax, grade in zip(axes, grade_order):
        entry = trajectories.get(grade)
        _bg(ax, occ)
        if entry is None:
            ax.set_title(f"{grade}\nnot generated", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        m = entry["metrics"]
        draw_trajectory(ax, entry["raw_path"], m["intervals"], lw=2.0, halo=True)
        _mark(ax, md["start"], "#2b6cff", ms=13)
        _label_point(ax, md["start"], "S", "#2b6cff", offset=(6, 5))
        _mark(ax, md["goal"], "#ff3b30", ms=13)
        _label_point(ax, md["goal"], "G", "#ff3b30", offset=(6, 5))
        ax.set_title(
            f"{grade}   ({entry['generator']})\n"
            f"collision_ratio={m['collision_ratio']:.3f}  intervals={m['collision_interval_count']}  "
            f"max_iv={m['max_collision_interval_length']:.1f}\nrho={m['rho']:.2f}  L={m['path_length']:.0f}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        Line2D([], [], color=COL_FREE, lw=2, label="free segment"),
        Line2D([], [], color=COL_COLLIDE, lw=2, label="collision segment"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{md['scene_id']}  {md['type']} {md['type_name']}   |   graded trajectories (raw, unrepaired)",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# §8.2b collision intervals of the worst trajectory
# --------------------------------------------------------------------------- #
def figure_collision_intervals(out_path, occ, md, grade, entry) -> None:
    from guidance_stage1.grid import Grid
    from guidance_stage2.collision import traversed_obstacle_cells

    raw = entry["raw_path"]
    intervals = entry["metrics"]["intervals"]
    grid = Grid(occ)
    crossed = traversed_obstacle_cells(grid, raw, min_penetration=0.05)
    fig, ax = plt.subplots(figsize=(7.6, 7.3), dpi=110)
    fig.patch.set_facecolor(BG)
    _bg(ax, occ)
    draw_traversed_cells(ax, crossed, alpha=0.5)
    draw_trajectory(ax, raw, intervals, lw=2.0)
    for iv in intervals:
        for key, color, marker in (("entry_cell", COL_BRIDGE, "s"), ("exit_cell", "#9467bd", "D")):
            cell = iv[key]
            ax.plot(cell[1], cell[0], marker, mfc=color, mec="white", mew=0.8, ms=6.5, zorder=8)
        mid = 0.5 * (np.array(iv["entry_point_rc"]) + np.array(iv["exit_point_rc"]))
        ax.annotate(
            f"#{iv['index']}  u[{iv['u_in']:.2f},{iv['u_out']:.2f}]",
            (mid[1], mid[0]),
            fontsize=6.5,
            color="black",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", alpha=0.9),
            zorder=9,
        )
    _mark(ax, md["start"], "#2b6cff", ms=11)
    _label_point(ax, md["start"], "S", "#2b6cff")
    _mark(ax, md["goal"], "#ff3b30", ms=11)
    _label_point(ax, md["goal"], "G", "#ff3b30")
    m = entry["metrics"]
    fig.suptitle(
        f"{md['scene_id']}  ·  {md['type']} {md['type_name']}  ·  {grade}  ·  collision intervals",
        fontsize=12, fontweight="bold", y=0.985,
    )
    ax.set_title(
        f"collision_ratio={m['collision_ratio']:.3f}   intervals={m['collision_interval_count']}   "
        f"L_col={m['collision_length']:.1f} / L={m['path_length']:.1f}   rho={m['rho']:.2f}",
        fontsize=9.4,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        Line2D([], [], color=COL_FREE, lw=2, label="free segment"),
        Line2D([], [], color=COL_COLLIDE, lw=2, label="segment inside an obstacle"),
        Line2D([], [], marker="s", ls="", mfc=COL_COLLIDE, mec="white", alpha=0.6,
               label=f"obstacle cell crossed ({len(crossed)})"),
        Line2D([], [], marker="s", ls="", mfc=COL_BRIDGE, mec="white", label="entry anchor"),
        Line2D([], [], marker="D", ls="", mfc="#9467bd", mec="white", label="exit anchor"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5,
              frameon=False, fontsize=8)
    fig.text(0.5, 0.005,
             "red patches are the obstacle cells the trajectory actually passes through "
             "(black = obstacle, light = free space)",
             ha="center", fontsize=8.2, color="0.25")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# §8.3 the three source-handling methods on one trajectory
# --------------------------------------------------------------------------- #
def figure_source_methods(out_path, occ, md, grade, entry, sources: dict, cfg) -> None:
    raw = entry["raw_path"]
    intervals = entry["metrics"]["intervals"]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.3), dpi=110)
    fig.patch.set_facecolor(BG)
    titles = {
        "M0": "M0  drop obstacle samples\n(dropped samples: red x)",
        "M1": "M1  project to nearest free cell\n(orange dot + line = projection, max r=%d)" % cfg.projection_radius,
        "M2": "M2  local free-space bridging\n(green = bridge, partial support path)",
    }
    for ax, method in zip(axes, ("M0", "M1", "M2")):
        _bg(ax, occ)
        src = sources[method]
        if method == "M0":
            from guidance_stage2.source_methods import sample_raw_trajectory

            sampled = sample_raw_trajectory(raw, cfg.num_source_samples)
            kept = {(int(r), int(c)) for r, c in src["positions_rc"]}
            for (r, c) in sampled["cells"]:
                if (int(r), int(c)) not in kept:
                    ax.plot(c, r, "x", color=COL_COLLIDE, ms=7, mew=1.4, zorder=6)
        if method == "M1":
            from guidance_stage2.source_methods import sample_raw_trajectory

            sampled = sample_raw_trajectory(raw, cfg.num_source_samples)
            proj = src["positions_rc"]
            for (r, c), (pr, pc) in zip(sampled["cells"], proj):
                if (int(r), int(c)) != (int(pr), int(pc)):
                    ax.plot([c, pc], [r, pr], "-", color=COL_PROJECT, lw=0.8, zorder=6)
                    ax.plot(pc, pr, "o", color=COL_PROJECT, ms=3.2, zorder=6)
        if method == "M2" and src.get("support_path") is not None:
            support = src["support_path"]
            ax.plot(support[:, 1], support[:, 0], "-", color=COL_BRIDGE, lw=1.4, alpha=0.95, zorder=5)
        draw_trajectory(ax, raw, intervals, lw=2.2, alpha=1.0, halo=True)
        positions = np.asarray(src["positions_rc"]).reshape(-1, 2)
        if len(positions):
            origin = np.asarray(src["origin"])
            for label, color in (("original_free", COL_FREE), ("local_bridge", COL_BRIDGE), ("projected", COL_PROJECT)):
                sel = origin == label
                if sel.any():
                    ax.scatter(positions[sel, 1], positions[sel, 0], s=9, c=color, edgecolors="white",
                               linewidths=0.35, alpha=0.85, zorder=6)
        _mark(ax, md["start"], "#2b6cff", ms=13)
        _label_point(ax, md["start"], "S", "#2b6cff", offset=(6, 5))
        _mark(ax, md["goal"], "#ff3b30", ms=13)
        _label_point(ax, md["goal"], "G", "#ff3b30", offset=(6, 5))
        sm = src["metadata"]
        ax.set_title(
            f"{titles[method]}\nsources={sm['num_valid_sources']}/{sm['num_raw_samples']}  "
            f"dup={len(positions) - len({(int(r), int(c)) for r, c in positions})}  "
            f"invalid={sm['num_invalid_sources']}",
            fontsize=8.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{md['scene_id']} {md['type']}  |  {grade}  same raw trajectory, three handling methods "
        f"(collision_ratio={entry['metrics']['collision_ratio']:.3f})",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# §8.4 field comparison (methods x grades) and M2 weight ablation
# --------------------------------------------------------------------------- #
def figure_field_methods(out_path, occ, md, grade_order, trajectories, fields_by_case) -> None:
    rows = ("raw", "M0", "M1", "M2")
    n_cols = len(grade_order)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(4.3 * n_cols, 4.45 * len(rows)), dpi=100)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_2d(axes)
    image = None
    for col, grade in enumerate(grade_order):
        entry = trajectories.get(grade)
        for row, label in enumerate(rows):
            ax = axes[row, col]
            _bg(ax, occ)
            if entry is None:
                ax.set_title(f"{grade}: not generated", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            if label == "raw":
                draw_trajectory(ax, entry["raw_path"], entry["metrics"]["intervals"], lw=1.5)
                _mark(ax, md["start"], "#2b6cff", ms=5)
                _mark(ax, md["goal"], "#ff3b30", ms=5)
                ax.set_title(
                    f"{grade}  raw  collision={entry['metrics']['collision_ratio']:.3f}" if row == 0 else "",
                    fontsize=8,
                )
            else:
                case = fields_by_case.get((grade, label, "linear"))
                if case is None:
                    ax.set_title(f"{label}: no sources", fontsize=8)
                else:
                    fm = case["metrics"]
                    image = _field_panel(
                        ax, occ, case["field"],
                        f"{label}+linear  dF={'--' if fm['delta_F'] is None else format(fm['delta_F'], '+.2f')}  "
                        f"src={fm['num_sources']}",
                        "linear", raw=entry["raw_path"], intervals=entry["metrics"]["intervals"],
                        start=md.get("start"), goal=md.get("goal"),
                    )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
    if image is not None:
        cb = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
        cb.set_label("F(x)  (linear weights, clipped)", fontsize=9)
    fig.suptitle(
        f"{md['scene_id']} {md['type']}  |  source method vs. field (linear weights, lambda fixed)"
        f"   rows: raw / M0 / M1 / M2",
        fontsize=11, y=0.998,
    )
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def figure_field_difference(out_path, occ, md, grade_order, trajectories, cases, ref_field) -> None:
    """F_method - F_reference: the panel where M0/M1/M2 actually differ visibly.

    The absolute fields of the three methods look alike at a glance (the global ramp
    dominates), so the deviation from the collision-free reference is what has to be read.
    """
    methods = ("M0", "M1", "M2")
    n_cols = len(grade_order)
    fig, axes = plt.subplots(3, n_cols, figsize=(4.3 * n_cols, 4.35 * 3), dpi=100)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_2d(axes)
    diffs = {}
    for grade in grade_order:
        for method in methods:
            case = cases.get((grade, method, "linear"))
            if case is None:
                continue
            diffs[(grade, method)] = case["field"] - ref_field
    finite = np.concatenate([np.abs(d[np.isfinite(d)]).ravel() for d in diffs.values()]) if diffs else np.zeros(1)
    vmax = float(np.percentile(finite, 98)) if finite.size else 0.05
    vmax = max(vmax, 0.02)
    image = None
    for col, grade in enumerate(grade_order):
        entry = trajectories.get(grade)
        for row, method in enumerate(methods):
            ax = axes[row, col]
            _bg(ax, occ)
            d = diffs.get((grade, method))
            if d is None:
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            image = ax.imshow(np.where(np.isfinite(d), d, np.nan), cmap="coolwarm",
                              vmin=-vmax, vmax=vmax, interpolation="nearest", alpha=0.95)
            if entry is not None:
                draw_trajectory(ax, entry["raw_path"], entry["metrics"]["intervals"], lw=1.0, alpha=0.9)
            fm = cases[(grade, method, "linear")]["metrics"]
            ax.set_title(f"{method} - ref   MAE={fm.get('mae_all_free', float('nan')):.3f}\n"
                         f"collision={entry['metrics']['collision_ratio']:.3f}" if entry else method, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(method, fontsize=10)
    if image is not None:
        cb = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
        cb.set_label("F(method) - F(collision-free reference)  [linear weights]", fontsize=9)
    fig.suptitle(
        f"{md['scene_id']} {md['type']}  |  deviation from the collision-free reference field "
        f"(red = weaker than reference, blue = stronger)   vmax={vmax:.2f}",
        fontsize=11, y=0.998,
    )
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def figure_single_field(out_path, occ, field, title, subtitle, kind, raw=None, intervals=None,
                        support=None, sources=None, source_progress=None, start=None, goal=None,
                        vmin=None, vmax=None, caption=None) -> None:
    """One field, full size: the deliverable image for a single (trajectory, method, weighting).

    Obstacles are dark (the convention used by every figure here), the field is drawn only on
    free space, the raw trajectory keeps its red collision segments and the M2 support path is
    green.  Start and goal are marked and labelled, and the figure carries a bold title on top
    plus a caption line underneath.
    """
    if vmin is None or vmax is None:
        vmin, vmax = display_range(kind)
    fig, ax = plt.subplots(figsize=(8.6, 7.9), dpi=110)
    fig.patch.set_facecolor(BG)
    _bg(ax, occ)
    image = ax.imshow(
        np.where(np.isfinite(field), np.clip(field, vmin, vmax), np.nan),
        cmap=FIELD_CMAP, vmin=vmin, vmax=vmax, interpolation="nearest", alpha=1.0,
    )
    # obstacles: dark (same convention as every other figure), with a light boundary so
    # they stay visible where the field itself is dark purple
    ax.imshow(np.where(occ == 1, OBS_DARK, np.nan), cmap="gray", vmin=0.0, vmax=1.0,
              interpolation="nearest", alpha=1.0)
    ax.contour(np.asarray(occ, dtype=float), levels=[0.5], colors="0.78", linewidths=0.45, alpha=0.9)
    if support is not None and len(support) > 1:
        ax.plot(support[:, 1], support[:, 0], "-", color="black", lw=3.0, alpha=0.45, zorder=5)
        ax.plot(support[:, 1], support[:, 0], "-", color=COL_BRIDGE, lw=1.7, alpha=1.0, zorder=6)
    if raw is not None:
        draw_trajectory(ax, raw, intervals or [], lw=1.8, alpha=1.0, halo=True)
    if sources is not None and len(sources):
        src = np.asarray(sources).reshape(-1, 2)
        if source_progress is not None and len(source_progress) == len(src):
            ax.scatter(src[:, 1], src[:, 0], c=source_progress, cmap=OWNER_CMAP, vmin=0.0, vmax=1.0,
                       s=22, edgecolors="white", linewidths=0.5, zorder=7)
        else:
            ax.scatter(src[:, 1], src[:, 0], c="white", edgecolors="black", s=10, zorder=7)
    if start is not None:
        _mark(ax, start, "#2b6cff", ms=15)
        _label_point(ax, start, "S", "#2b6cff")
    if goal is not None:
        _mark(ax, goal, "#ff3b30", ms=15)
        _label_point(ax, goal, "G", "#ff3b30")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(subtitle, fontsize=9.4)
    fig.suptitle(title, fontsize=12.6, fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.012,
        caption or (f"F(x) = max_i [ w(u_i) - lambda * d_free(x, s_i) ]     lambda = {0.02}     "
                    "black = obstacle (no field)     dot colour = original trajectory progress u"),
        ha="center", fontsize=8.2, color="0.25",
    )
    cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(f"F(x)  [{kind} weights, clipped to {vmin:.2f} .. {vmax:.2f}]", fontsize=9)
    handles = [
        Line2D([], [], color=COL_FREE, lw=2, label="free segment"),
        Line2D([], [], color=COL_COLLIDE, lw=2, label="in obstacle"),
        Line2D([], [], color=COL_BRIDGE, lw=2, label="M2 bridge / support path"),
        Line2D([], [], marker="s", ls="", mfc="#1a1a1a", mec="0.8", label="obstacle"),
        Line2D([], [], marker="o", ls="", mfc="#2b6cff", mec="white", label="S start"),
        Line2D([], [], marker="o", ls="", mfc="#ff3b30", mec="white", label="G goal"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=6,
              fontsize=7.5, frameon=False)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.90, bottom=0.12)
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def figure_weight_ablation(out_path, occ, md, grade_order, trajectories, fields_by_case) -> None:
    kinds = ("uniform", "linear", "quadratic")
    n_cols = len(grade_order)
    fig, axes = plt.subplots(len(kinds), n_cols, figsize=(4.3 * n_cols, 4.45 * len(kinds)), dpi=100)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_2d(axes)
    image = None
    for col, grade in enumerate(grade_order):
        entry = trajectories.get(grade)
        for row, kind in enumerate(kinds):
            ax = axes[row, col]
            _bg(ax, occ)
            case = fields_by_case.get((grade, "M2", kind)) if entry is not None else None
            if case is None:
                ax.set_title("no field", fontsize=8)
            else:
                fm = case["metrics"]
                image = _field_panel(
                    ax, occ, case["field"],
                    f"M2+{kind}  dF={'--' if fm['delta_F'] is None else format(fm['delta_F'], '+.2f')}  "
                    f"owners={fm['owner_distinct']}",
                    kind, raw=entry["raw_path"], intervals=entry["metrics"]["intervals"],
                    support=case.get("support_path"),
                    start=md.get("start"), goal=md.get("goal"),
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(kind, fontsize=10)
    if image is not None:
        cb = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
        cb.set_label("F(x)  (M2 sources, clipped)", fontsize=9)
    fig.suptitle(
        f"{md['scene_id']} {md['type']}  |  M2 weight ablation (uniform / linear / quadratic)",
        fontsize=11, y=0.998,
    )
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# §8.5 dominant-source partition
# --------------------------------------------------------------------------- #
def figure_owners(out_path, occ, md, grade, entry, fields_by_case) -> None:
    methods = ("M0", "M1", "M2")
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 5.1), dpi=110)
    fig.patch.set_facecolor(BG)
    image = None
    for ax, method in zip(axes, methods):
        _bg(ax, occ)
        case = fields_by_case.get((grade, method, "linear"))
        if case is None:
            ax.set_title(f"{method}: no field", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        owner = case["owner"]
        progress = case["source_progress"]
        # colour by the *progress* of the dominating source, not by its list index
        owner_progress = np.where(owner >= 0, np.interp(owner, np.arange(len(progress)), progress)
                                 if len(progress) else -1.0, np.nan)
        image = ax.imshow(owner_progress, cmap=OWNER_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest", alpha=0.92)
        if case.get("support_path") is not None:
            sup = case["support_path"]
            ax.plot(sup[:, 1], sup[:, 0], "-", color=COL_BRIDGE, lw=1.2, alpha=0.95, zorder=5)
        raw = entry["raw_path"]
        intervals = entry["metrics"]["intervals"]
        segs, colors = _colored_segments(raw, intervals)
        for seg, col in zip(segs, colors):
            if col == COL_COLLIDE:
                ax.plot(seg[:, 1], seg[:, 0], "--", color=COL_COLLIDE, lw=1.0, zorder=6)
        _mark(ax, md["start"], "#2b6cff", ms=11)
        _label_point(ax, md["start"], "S", "#2b6cff", offset=(5, 4))
        _mark(ax, md["goal"], "#ff3b30", ms=11)
        _label_point(ax, md["goal"], "G", "#ff3b30", offset=(5, 4))
        fm = case["metrics"]
        ax.set_title(
            f"{method}  owners={fm['owner_distinct']}  entropy={fm['owner_entropy']:.2f}  "
            f"dominant={fm['owner_dominant_fraction']:.2f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    if image is not None:
        cb = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        cb.set_label("original progress u of the dominating source", fontsize=9)
    fig.suptitle(
        f"{md['scene_id']} {md['type']}  |  {grade}: dominant-source partition "
        f"(red dashed = raw collision, green = M2 support path)",
        fontsize=11, y=0.98,
    )
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# §8.6 progress profiles
# --------------------------------------------------------------------------- #
def figure_progress_profiles(out_path, md, grade_order, trajectories, sources_by_case, fields_by_case) -> None:
    n = len(grade_order)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0), dpi=110, sharey=True)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes)
    colors = {"M0": "#4c72b0", "M1": "#dd8452", "M2": "#55a868"}
    for ax, grade in zip(axes, grade_order):
        entry = trajectories.get(grade)
        if entry is None:
            ax.set_title(f"{grade}: not generated", fontsize=9)
            continue
        m = entry["metrics"]
        for iv in m["intervals"]:
            ax.axvspan(iv["u_in"], iv["u_out"], color=COL_COLLIDE, alpha=0.16, lw=0)
        for method in ("M0", "M1", "M2"):
            src = sources_by_case.get((grade, method))
            if src is None:
                continue
            prog = np.asarray(src["progress"], dtype=float)
            field = fields_by_case[(grade, method, "linear")]["field"]
            vals = np.array([field[int(r), int(c)] for r, c in np.asarray(src["positions_rc"])])
            order = np.argsort(prog)
            ax.plot(prog[order], vals[order], "o-", ms=3.0, lw=1.2, color=colors[method], label=method)
            if method == "M2":
                for b in src["metadata"].get("bridges", []):
                    ax.axvspan(b["u_in"], b["u_out"], color=COL_BRIDGE, alpha=0.13, lw=0)
        ax.plot([0, 1], [0, 1], ":", color="gray", lw=1.0, label="ideal w=u")
        ax.set_title(f"{grade}\nratio={m['collision_ratio']:.3f}  intervals={m['collision_interval_count']}", fontsize=8.5)
        ax.set_xlabel("original progress u (not renormalised)")
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
    axes[0].set_ylabel("F(s_i)  field value at the source")
    axes[-1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(
        f"{md['scene_id']} {md['type']}  |  progress profile per source method "
        f"(red band = raw collision interval, green band = M2 bridge)",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# summary figures
# --------------------------------------------------------------------------- #
def figure_collision_severity(out_path, records) -> None:
    grades = ("feasible", "mild_collision", "moderate_collision", "severe_collision")
    types = ("A", "B", "C", "D")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), dpi=110)
    fig.patch.set_facecolor(BG)
    ax = axes[0]
    for k, grade in enumerate(grades):
        vals = [r["collision_ratio"] for r in records if r["grade"] == grade and r["collision_ratio"] is not None]
        if not vals:
            continue
        ax.scatter([k + np.random.uniform(-0.18, 0.18) for _ in vals], vals, s=18,
                   color=GRADE_COLORS[grade], alpha=0.8, edgecolors="white", linewidths=0.3)
        ax.plot([k - 0.28, k + 0.28], [np.median(vals)] * 2, color="black", lw=2)
    ax.set_xticks(range(len(grades)))
    ax.set_xticklabels([g.replace("_collision", "") for g in grades], fontsize=8.5)
    ax.set_ylabel("measured collision_ratio")
    ax.set_title("collision ratio per grade (black bar = median)", fontsize=9.5)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for k, grade in enumerate(grades[1:], start=1):
        for j, typ in enumerate(types):
            vals = [r["collision_ratio"] for r in records if r["grade"] == grade and r["map_type"] == typ and r["collision_ratio"] is not None]
            if not vals:
                continue
            ax.scatter([j + 0.2 * (k - 1) - 0.2 for _ in vals], vals, s=14, color=GRADE_COLORS[grade], alpha=0.75)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types)
    ax.set_xlabel("map family")
    ax.set_title("collision ratio by map family (colour = grade)", fontsize=9.5)
    ax.grid(alpha=0.3, axis="y")
    handles = [Line2D([], [], marker="o", ls="", color=GRADE_COLORS[g], label=g) for g in grades]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right")
    fig.suptitle("Stage 2A: measured collision severity", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


def figure_method_comparison(out_path, records) -> None:
    """Field-level effect of M0/M1/M2 per grade (linear weights)."""
    grades = ("feasible", "mild_collision", "moderate_collision", "severe_collision")
    methods = ("M0", "M1", "M2")
    colors = {"M0": "#4c72b0", "M1": "#dd8452", "M2": "#55a868"}
    panels = (
        ("delta_F", "delta_F = F_goal - F_start"),
        ("owner_entropy", "normalised owner entropy"),
        ("progress_covered_fraction", "progress coverage of the sources"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), dpi=110)
    fig.patch.set_facecolor(BG)
    for ax, (key, label) in zip(axes, panels):
        for m_i, method in enumerate(methods):
            means, errs, xs = [], [], []
            for g_i, grade in enumerate(grades):
                vals = [
                    r[key]
                    for r in records
                    if r["method"] == method and r["grade"] == grade and r[key] is not None and r["weight"] == "linear"
                ]
                if not vals:
                    continue
                means.append(float(np.mean(vals)))
                errs.append(float(np.std(vals)))
                xs.append(g_i + (m_i - 1) * 0.26)
            if xs:
                ax.bar(xs, means, yerr=errs, width=0.24, color=colors[method], capsize=2,
                       label=method, error_kw={"lw": 0.8})
        ax.set_xticks(range(len(grades)))
        ax.set_xticklabels([g.replace("_collision", "") for g in grades], fontsize=8)
        ax.set_title(label, fontsize=9.5)
        ax.grid(alpha=0.3, axis="y")
        ax.axhline(0.0, color="black", lw=0.7)
    axes[0].legend(fontsize=8)
    fig.suptitle("Stage 2A: source method comparison (linear weights)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


def figure_field_metrics(out_path, records) -> None:
    """Weighting ablation on M2 sources plus the reference-agreement statistics."""
    kinds = ("uniform", "linear", "quadratic")
    colors = {"uniform": "#4c72b0", "linear": "#dd8452", "quadratic": "#55a868"}
    panels = (
        ("delta_F", "delta_F"),
        ("owner_distinct", "distinct owners"),
        ("corr_all_free", "Pearson corr vs. reference field"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), dpi=110)
    fig.patch.set_facecolor(BG)
    grades = ("feasible", "mild_collision", "moderate_collision", "severe_collision")
    for ax, (key, label) in zip(axes, panels):
        for m_i, kind in enumerate(kinds):
            means, xs = [], []
            for g_i, grade in enumerate(grades):
                vals = [
                    r[key]
                    for r in records
                    if r["method"] == "M2" and r["weight"] == kind and r["grade"] == grade and r[key] is not None
                ]
                if not vals:
                    continue
                means.append(float(np.mean(vals)))
                xs.append(g_i + (m_i - 1) * 0.26)
            if xs:
                ax.bar(xs, means, width=0.24, color=colors[kind], label=kind)
        ax.set_xticks(range(len(grades)))
        ax.set_xticklabels([g.replace("_collision", "") for g in grades], fontsize=8)
        ax.set_title(label, fontsize=9.5)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Stage 2A: M2 weighting ablation and agreement with the reference field", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
