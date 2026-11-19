"""Compact plotting helpers: occupancy, guidance field, search expansion.

Conventions used everywhere: obstacles dark, free space light / field colour, S blue with
the letter S, G red with the letter G, free trajectory segments blue, segments that pass
through obstacles red, M2 bridges green.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from .collision import arc_lengths, point_at_arc_length
from .config import field_clip_bounds

FIELD_CMAP = "viridis"
PROGRESS_CMAP = "plasma"
EXPANSION_CMAP = "plasma"
FREE_LIGHT = 0.93
OBS_DARK = 0.10
COL_FREE = "#1f77b4"
COL_COLLIDE = "#d62728"
COL_BRIDGE = "#2ca02c"
BG = "white"


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def draw_occupancy(ax, occupancy):
    ax.imshow(np.where(occupancy == 1, OBS_DARK, FREE_LIGHT), cmap="gray", vmin=0.0, vmax=1.0,
              interpolation="nearest")


def _colored_segments(raw, intervals):
    """Split the polyline exactly at collision boundaries: (segments, colours)."""
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
            segs.append(np.array([point_at_arc_length(raw, cum, a), point_at_arc_length(raw, cum, b)]))
            colors.append(COL_COLLIDE if inside else COL_FREE)
    return [s[:, ::-1] for s in segs], colors  # (x, y) = (col, row)


def draw_trajectory(ax, raw, intervals=(), lw: float = 1.8, halo: bool = True):
    """Trajectory in image coordinates (x = col, y = row), collision parts in red."""
    if raw is None or len(raw) < 2:
        return
    segs, colors = _colored_segments(np.asarray(raw, dtype=float), list(intervals))
    if not segs:
        return
    if halo:
        ax.add_collection(LineCollection(segs, colors="white", linewidths=lw + 2.2, alpha=0.9, zorder=4))
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw, capstyle="round", zorder=5))


def draw_endpoints(ax, start, goal, ms: int = 13):
    for cell, color, letter, dx in ((start, "#2b6cff", "S", 6), (goal, "#ff3b30", "G", 6)):
        if cell is None:
            continue
        ax.plot(cell[1], cell[0], "o", mfc=color, mec="white", mew=1.4, ms=ms, zorder=9)
        ax.annotate(letter, (cell[1], cell[0]), xytext=(dx, 5), textcoords="offset points",
                    fontsize=13, fontweight="bold", color="white", zorder=10,
                    path_effects=[pe.withStroke(linewidth=2.8, foreground="black")])


def draw_sources(ax, positions, progress=None, size: float = 11):
    if positions is None or len(positions) == 0:
        return
    p = np.asarray(positions).reshape(-1, 2)
    if progress is not None and len(progress) == len(p):
        ax.scatter(p[:, 1], p[:, 0], c=np.asarray(progress), cmap=PROGRESS_CMAP, vmin=0, vmax=1,
                   s=size, edgecolors="white", linewidths=0.3, zorder=7)
    else:
        ax.scatter(p[:, 1], p[:, 0], c="white", edgecolors="black", linewidths=0.3, s=size, zorder=7)


def draw_expansion(ax, grid, trace, size: float = 5, alpha: float = 0.85):
    if not trace:
        return
    rows, cols = np.divmod(np.asarray(trace, dtype=int), grid.width)
    ax.scatter(cols, rows, c=np.arange(len(rows)), cmap=EXPANSION_CMAP, s=size, marker="s",
               linewidths=0, alpha=alpha, zorder=6)


def draw_path(ax, grid, path, lw: float = 1.8):
    if not path:
        return
    rows, cols = np.divmod(np.asarray(path, dtype=int), grid.width)
    ax.plot(cols, rows, "-", color="black", lw=lw + 1.2, alpha=0.55, zorder=7)
    ax.plot(cols, rows, "-", color="#ff2d55", lw=lw, zorder=8)


def _finish(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def _field_image(ax, field, weight):
    lo, hi = field_clip_bounds(weight)
    return ax.imshow(np.where(np.isfinite(field), np.clip(field, lo, hi), np.nan), cmap=FIELD_CMAP,
                     vmin=lo, vmax=hi, interpolation="nearest", alpha=0.93)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_field(out_path, grid, field, weight: str, raw=None, intervals=(), sources=None,
                 progress=None, start=None, goal=None, title: str = "", caption: str = "") -> None:
    """One full-size field image with trajectory, sources and endpoints."""
    fig, ax = plt.subplots(figsize=(8.2, 7.6), dpi=110)
    fig.patch.set_facecolor(BG)
    draw_occupancy(ax, grid.occupancy)
    image = _field_image(ax, field, weight)
    ax.contour(np.asarray(grid.occupancy, dtype=float), levels=[0.5], colors="0.78", linewidths=0.45)
    draw_sources(ax, sources, progress)
    draw_trajectory(ax, raw, intervals)
    draw_endpoints(ax, start, goal)
    _finish(ax)
    if title:
        fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.98)
    if caption:
        fig.text(0.5, 0.012, caption, ha="center", fontsize=8.2, color="0.25")
    cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(f"F(x)  [{weight} weights]", fontsize=9)
    handles = [
        Line2D([], [], color=COL_FREE, lw=2, label="trajectory (free)"),
        Line2D([], [], color=COL_COLLIDE, lw=2, label="trajectory (in obstacle)"),
        Line2D([], [], marker="s", ls="", mfc="#1a1a1a", mec="0.8", label="obstacle"),
        Line2D([], [], marker="o", ls="", mfc="#2b6cff", mec="white", label="S"),
        Line2D([], [], marker="o", ls="", mfc="#ff3b30", mec="white", label="G"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5,
              frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.90, bottom=0.12)
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def figure_search(out_path, grid, field, weight: str, trace, path, start, goal,
                  title: str = "", raw=None, intervals=()) -> None:
    """Expansion trace + final path drawn on top of the guidance field."""
    fig, ax = plt.subplots(figsize=(8.0, 7.4), dpi=110)
    fig.patch.set_facecolor(BG)
    draw_occupancy(ax, grid.occupancy)
    image = _field_image(ax, field, weight) if field is not None else None
    draw_trajectory(ax, raw, intervals, lw=1.4)
    draw_expansion(ax, grid, trace)
    draw_path(ax, grid, path)
    draw_endpoints(ax, start, goal)
    _finish(ax)
    if title:
        fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.98)
    if image is not None:
        cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(f"F(x)  [{weight} weights]", fontsize=9)
    handles = [
        Line2D([], [], marker="s", ls="", mfc="orange", mec="none", label="expanded node"),
        Line2D([], [], color="#ff2d55", lw=2, label="final path"),
        Line2D([], [], color=COL_FREE, lw=2, label="raw trajectory"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
              frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.10)
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def figure_grid(out_path, panels, ncols: int = 3, figsize=None, suptitle: str = "") -> None:
    """Generic contact sheet: ``panels`` is a list of callables taking an axes."""
    nrows = int(np.ceil(len(panels) / ncols))
    figsize = figsize or (5.2 * ncols, 5.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=100)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, draw in zip(axes, panels):
        ax.axis("on")
        draw(ax)
        _finish(ax)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
