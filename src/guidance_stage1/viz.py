"""Figures for stage 1: per-trajectory panels, quality comparison, along-path profiles."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .fields import source_weights

FREE_LIGHT = 0.90
OBS_DARK = 0.12
FIELD_CMAP = "viridis"
SOURCE_CMAP = "plasma"
BG = "white"


def _raster_rc(nodes: list[int], width: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array([divmod(int(i), width) for i in nodes])
    return arr[:, 1], arr[:, 0]


def _draw_maze(ax, occupancy: np.ndarray, alpha: float = 1.0) -> None:
    ax.imshow(
        np.where(occupancy == 1, OBS_DARK, FREE_LIGHT),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        alpha=alpha,
    )


def _draw_path(ax, nodes, width, color="white", lw=1.3, halo=True) -> None:
    cs, rs = _raster_rc(nodes, width)
    if halo:
        ax.plot(cs, rs, "-", color="black", lw=lw + 1.2, alpha=0.35, solid_capstyle="round")
    ax.plot(cs, rs, "-", color=color, lw=lw, solid_capstyle="round")


def _draw_endpoints(ax, start, goal, size=9) -> None:
    ax.plot(start[1], start[0], "o", mfc="#2b6cff", mec="white", mew=1.0, ms=size, zorder=6)
    ax.plot(goal[1], goal[0], "o", mfc="#ff3b30", mec="white", mew=1.0, ms=size, zorder=6)


def _draw_sources(ax, samples: np.ndarray, weights: np.ndarray) -> None:
    sizes = 26.0 + 120.0 * (weights - weights.min()) / max(float(np.ptp(weights)), 1e-9)
    ax.scatter(
        samples[:, 1],
        samples[:, 0],
        s=sizes,
        c=weights,
        cmap=SOURCE_CMAP,
        vmin=float(min(weights.min(), 0.0)),
        vmax=float(max(weights.max(), 1.0)),
        edgecolors="white",
        linewidths=0.5,
        zorder=5,
    )


def weight_display_range(kind: str, m: int) -> tuple[float, float]:
    """Colour range that exposes the structure of one weighting.

    Far from the trajectory every field decays to a large negative number, and if that
    tail sets the scale the 0..1 ramp along the trajectory becomes invisible.  Each
    panel therefore shows the range where the mechanism actually lives and clips the
    rest (marked with a triangular end on the colourbar); raw extremes stay in the data.
    """
    w = source_weights(m, kind)
    if kind == "uniform":
        # equal weights: only the "how far does it reach" gradient is interesting
        return float(w.max() - 0.50), float(w.max())
    return float(w.min() - 0.10), float(w.max())


def _clip_for_display(field: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip(field, vmin, vmax)


def _colormap():
    cmap = plt.get_cmap(FIELD_CMAP).copy()
    cmap.set_bad(alpha=0.0)
    cmap.set_under("#3a3a3a")
    return cmap


def _field_panel(ax, field, occupancy, title, vmin, vmax, draw_path_nodes=None, width=None):
    _draw_maze(ax, occupancy, alpha=0.9)
    image = ax.imshow(
        np.where(np.isfinite(field), _clip_for_display(field, vmin, vmax), np.nan),
        cmap=_colormap(),
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        alpha=0.92,
    )
    if draw_path_nodes is not None:
        _draw_path(ax, draw_path_nodes, width, color="white", lw=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def figure_trajectory(
    out_path,
    scene_meta: dict,
    occupancy,
    grade: str,
    cand: dict,
    samples,
    weights: dict,
    fields: dict,
    cfg,
    display_ranges: dict | None = None,
) -> None:
    """The 4-panel stage-1 output: maze+trajectory, Field-A, Field-B, Field-B contours."""
    width = occupancy.shape[1]
    start = scene_meta["start"]
    goal = scene_meta["goal"]
    display_ranges = display_ranges or {
        kind: weight_display_range(kind, cfg.num_samples) for kind in ("uniform", "linear", "quadratic")
    }
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.3), dpi=110)
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    _draw_maze(ax, occupancy)
    _draw_path(ax, cand["nodes"], width, color="#c81e3c", lw=1.5)
    _draw_sources(ax, samples, weights["linear"])
    _draw_endpoints(ax, start, goal)
    ax.set_title("1. Maze + trajectory + 32 sources\n(marker size/colour = w_i)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    mappable = plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=SOURCE_CMAP)
    fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02, label="w_i (0 = start, 1 = goal)")

    for ax, kind, title in (
        (axes[1], "uniform", "2. Field-A  uniform  w_i = 1"),
        (axes[2], "linear", "3. Field-B  linear  w_i = u_i"),
        (axes[3], "linear", "4. Field-B contours"),
    ):
        lims = display_ranges[kind]
        if title.endswith("contours"):
            _draw_maze(ax, occupancy, alpha=1.0)
            clipped = _clip_for_display(fields["linear"], *lims)
            back = ax.imshow(
                np.where(np.isfinite(fields["linear"]), clipped, np.nan),
                cmap=_colormap(),
                vmin=lims[0],
                vmax=lims[1],
                interpolation="nearest",
                alpha=0.62,
            )
            levels = np.linspace(lims[0], lims[1], 13)
            ax.contour(
                np.ma.masked_invalid(_clip_for_display(fields["linear"], *lims)),
                levels=levels,
                colors="black",
                linewidths=0.8,
                alpha=0.75,
            )
            _draw_path(ax, cand["nodes"], width, color="#ff2d55", lw=1.3)
            _draw_endpoints(ax, start, goal, size=8)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(back, ax=ax, fraction=0.046, pad=0.02, label="F(x)  with iso-contours")
            continue
        image = _field_panel(ax, fields[kind], occupancy, title, *lims, draw_path_nodes=cand["nodes"], width=width)
        _draw_endpoints(ax, start, goal, size=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label="F(x)")

    fig.suptitle(
        f"{scene_meta['scene_id']}  type {scene_meta['type']} ({scene_meta['type_name']})   |   "
        f"grade {grade}   rho = {cand['ratio']:.3f}   L = {cand['length']}   "
        f"(L* = {scene_meta['stats']['geodesic_sg']})   |   lambda = {cfg.lam}",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


def figure_quality_comparison(out_path, scene_meta: dict, occupancy, grades, per_grade: dict, cfg) -> None:
    """Same map, four trajectory qualities (columns) x three weightings (rows)."""
    width = occupancy.shape[1]
    names = [g for g in grades if g in per_grade]
    kinds = (("uniform", "Field-A  uniform  w_i = 1"), ("linear", "Field-B  linear  w_i = u_i"),
             ("quadratic", "Field-C  quadratic  w_i = u_i^2"))
    fig, axes = plt.subplots(3, len(names), figsize=(4.6 * len(names), 13.6), dpi=110)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_2d(axes)
    start, goal = scene_meta["start"], scene_meta["goal"]
    images = {}
    for row, (kind, kind_label) in enumerate(kinds):
        lims = weight_display_range(kind, cfg.num_samples)
        for col, grade in enumerate(names):
            entry = per_grade[grade]
            ax = axes[row, col]
            image = _field_panel(
                ax,
                entry["fields"][kind],
                occupancy,
                (f"{grade}   rho = {entry['cand']['ratio']:.3f}   L = {entry['cand']['length']}\n"
                 f"dF = {entry['metrics'][kind]['delta_F']:+.3f}   owners = {entry['metrics'][kind]['owner_distinct']}")
                if row == 0
                else f"dF = {entry['metrics'][kind]['delta_F']:+.3f}   owners = {entry['metrics'][kind]['owner_distinct']}",
                *lims,
                draw_path_nodes=entry["cand"]["nodes"],
                width=width,
            )
            _draw_endpoints(ax, start, goal, size=7)
            if col == 0:
                ax.set_ylabel(kind_label, fontsize=9.5)
            images[kind] = image
        cb = fig.colorbar(images[kind], ax=axes[row, :].tolist(), fraction=0.02, pad=0.01)
        cb.set_label(f"F(x)  {kind}", fontsize=9)
    fig.suptitle(
        f"{scene_meta['scene_id']}  type {scene_meta['type']} ({scene_meta['type_name']})   |   "
        f"trajectory quality vs. resulting global guidance field   |   lambda = {cfg.lam}",
        fontsize=11,
        y=0.996,
    )
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def figure_owners(out_path, scene_meta: dict, occupancy, grades, per_grade: dict, cfg) -> None:
    """Which trajectory sample dominates each cell (the field's global structure)."""
    width = occupancy.shape[1]
    names = [g for g in grades if g in per_grade]
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 5.1), dpi=110)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes)
    start, goal = scene_meta["start"], scene_meta["goal"]
    image = None
    for ax, grade in zip(axes, names):
        entry = per_grade[grade]
        _draw_maze(ax, occupancy, alpha=0.9)
        owner = entry["owners"]["linear"]
        image = ax.imshow(
            np.where(owner >= 0, owner, np.nan),
            cmap=SOURCE_CMAP,
            vmin=0,
            vmax=cfg.num_samples - 1,
            interpolation="nearest",
            alpha=0.92,
        )
        _draw_path(ax, entry["cand"]["nodes"], width, color="white", lw=0.9)
        _draw_endpoints(ax, start, goal, size=7)
        ax.set_title(
            f"{grade}   rho = {entry['cand']['ratio']:.3f}\n"
            f"distinct owners = {entry['metrics']['linear']['owner_distinct']} / {cfg.num_samples}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    if image is not None:
        cb = fig.colorbar(image, ax=axes.tolist(), fraction=0.025, pad=0.01)
        cb.set_label(f"dominant sample index (0 = start .. {cfg.num_samples - 1} = goal)", fontsize=9)
    fig.suptitle(
        f"{scene_meta['scene_id']}  type {scene_meta['type']}   |   dominant-source partition of the map "
        f"(Field-B weights)",
        fontsize=11,
        y=0.98,
    )
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def figure_profiles(out_path, scene_meta: dict, grades, per_grade: dict, cfg) -> None:
    """Field value along the trajectory for the three weightings, one panel per grade."""
    names = [g for g in grades if g in per_grade]
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.9), dpi=110, sharey=True)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes)
    u = np.linspace(0.0, 1.0, cfg.num_samples)
    styles = {
        "uniform": ("#4c72b0", "-", "Field-A uniform"),
        "linear": ("#dd8452", "-", "Field-B linear"),
        "quadratic": ("#55a868", "--", "Field-C quadratic"),
    }
    for ax, grade in zip(axes, names):
        entry = per_grade[grade]
        for kind, (color, ls, label) in styles.items():
            profile = entry["profiles"][kind]
            ax.plot(u, profile, ls, color=color, lw=1.8, label=label)
        ax.plot(u, u, ":", color="gray", lw=1.0, label="ideal ramp w = u")
        ax.set_title(
            f"{grade}  rho={entry['cand']['ratio']:.2f}  "
            f"dF_B={entry['metrics']['linear']['delta_F']:+.2f}",
            fontsize=9,
        )
        ax.set_xlabel("trajectory progress u")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.08)
    axes[0].set_ylabel("F(s_i)  field value at source i")
    axes[-1].legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.suptitle(
        f"{scene_meta['scene_id']}  type {scene_meta['type']}   |   along-trajectory field profile per weighting",
        fontsize=11,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


GRADE_COLORS = {
    "excellent": "#1f77b4",
    "good": "#2ca02c",
    "medium": "#ff7f0e",
    "poor": "#d62728",
}


def _draw_all_trajectories(ax, per_grade: dict, width: int, labels: bool = True, lw: float = 1.5) -> None:
    for grade in ("poor", "medium", "good", "excellent"):  # best drawn last, on top
        entry = per_grade.get(grade)
        if entry is None:
            continue
        cand = entry["cand"]
        label = f"{grade}  rho={cand['ratio']:.2f}  L={cand['length']}" if labels else None
        _draw_path(ax, cand["nodes"], width, color=GRADE_COLORS[grade], lw=lw, halo=True)
        if label:
            ax.plot([], [], "-", color=GRADE_COLORS[grade], lw=lw + 0.6, label=label)


def figure_trajectories(out_path, scene_meta: dict, occupancy, grades, per_grade: dict, cfg) -> None:
    """All trajectory grades of one map on a single panel (the 'working conditions' view)."""
    width = occupancy.shape[1]
    fig, ax = plt.subplots(figsize=(7.4, 7.0), dpi=110)
    fig.patch.set_facecolor(BG)
    _draw_maze(ax, occupancy)
    _draw_all_trajectories(ax, per_grade, width)
    _draw_endpoints(ax, scene_meta["start"], scene_meta["goal"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    st = scene_meta["stats"]
    ax.set_title(
        f"{scene_meta['scene_id']}  type {scene_meta['type']} ({scene_meta['type_name']})\n"
        f"L* = {st['geodesic_sg']}   free = {st['free_fraction']:.2f}   "
        f"pockets(detour>=10) = {st['pocket_cells_detour10']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def figure_scene_gallery(out_path, gallery: list[dict], cfg) -> None:
    """Contact sheet: every generated map with its four trajectory grades."""
    cols = max(1, cfg.instances_per_type)
    rows = len(gallery) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.9 * cols, 4.15 * rows), dpi=110)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, item in zip(axes.ravel(), gallery):
        ax.axis("on")
        width = item["occupancy"].shape[1]
        _draw_maze(ax, item["occupancy"])
        _draw_all_trajectories(ax, item["per_grade"], width, labels=False, lw=1.2)
        _draw_endpoints(ax, item["meta"]["start"], item["meta"]["goal"], size=6)
        st = item["meta"]["stats"]
        ax.set_title(
            f"{item['meta']['scene_id']}  type {item['meta']['type']}\n"
            f"L*={st['geodesic_sg']}  free={st['free_fraction']:.2f}  "
            f"rho={item['meta']['stats']['rho_max_in_pool']:.2f}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        plt.Line2D([], [], color=GRADE_COLORS[g], lw=2, label=g) for g in ("excellent", "good", "medium", "poor")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Generated scenes and their trajectory grades   |   rows = scene family A/B/C/D   |   lambda = {cfg.lam}",
        fontsize=12,
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def figure_summary(out_path, records) -> None:
    """Aggregate view: delta_F, owner diversity and monotonicity by grade and weighting."""
    grades = ["excellent", "good", "medium", "poor"]
    kinds = ["uniform", "linear", "quadratic"]
    colors = {"uniform": "#4c72b0", "linear": "#dd8452", "quadratic": "#55a868"}
    metrics = (
        ("delta_F", "delta_F = F_goal - F_start"),
        ("owner_entropy", "normalised owner entropy (global spread of influence)"),
        ("monotone_fraction", "monotone fraction of F(s_i) along the trajectory"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), dpi=110)
    fig.patch.set_facecolor(BG)
    for ax, (key, label) in zip(axes, metrics):
        means, errs, xs, labels, bar_colors = [], [], [], [], []
        pos = 0
        for gi, grade in enumerate(grades):
            for kind in kinds:
                vals = [
                    rec["metrics"][kind][key]
                    for rec in records
                    if rec["grade"] == grade and kind in rec["metrics"]
                ]
                vals = [v for v in vals if np.isfinite(v)]
                if not vals:
                    continue
                means.append(float(np.mean(vals)))
                errs.append(float(np.std(vals)))
                xs.append(pos)
                labels.append(f"{grade[:3]}\n{kind[:3]}")
                bar_colors.append(colors[kind])
                pos += 1
            pos += 0.8
        ax.bar(xs, means, yerr=errs, color=bar_colors, capsize=2.5, width=0.75)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_title(label, fontsize=9.5)
        ax.grid(alpha=0.3, axis="y")
        ax.axhline(0.0, color="black", lw=0.8)
    fig.suptitle("Stage-1 aggregate: 20 maps x 4 grades x 3 weightings", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
