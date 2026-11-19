"""Figures for the search stage: expansion overlays, method comparison, metric summaries."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from guidance_stage2.visualization import (
    COL_COLLIDE,
    COL_FREE,
    FIELD_CMAP,
    OBS_DARK,
    display_range,
)
from guidance_stage3.config import METHOD_LABELS

BG = "white"
EXPANSION_CMAP = "plasma"


def _background(ax, occ):
    ax.imshow(np.where(occ == 1, OBS_DARK, 0.93), cmap="gray", vmin=0, vmax=1, interpolation="nearest")


def _field(ax, occ, field, weight):
    lo, hi = display_range(weight)
    _background(ax, occ)
    return ax.imshow(np.where(np.isfinite(field), np.clip(field, lo, hi), np.nan),
                     cmap=FIELD_CMAP, vmin=lo, vmax=hi, interpolation="nearest", alpha=0.92)


def _trajectory(ax, raw):
    if raw is None or len(raw) < 2:
        return
    segs = [np.array([[p[1], p[0]] for p in raw[j:j + 2]]) for j in range(len(raw) - 1)]
    ax.add_collection(LineCollection(segs, colors="white", linewidths=3.4, alpha=0.85, zorder=4))
    ax.add_collection(LineCollection(segs, colors=COL_FREE, linewidths=1.6, alpha=0.95, zorder=5))


def _expansion(ax, grid, trace, width):
    if not trace:
        return
    rows, cols = np.divmod(np.asarray(trace, dtype=int), width)
    order = np.arange(len(rows))
    ax.scatter(cols, rows, c=order, cmap=EXPANSION_CMAP, s=5, marker="s", linewidths=0,
               alpha=0.85, zorder=6)


def _path(ax, grid, path):
    if not path:
        return
    rows, cols = np.divmod(np.asarray(path, dtype=int), grid.width)
    ax.plot(cols, rows, "-", color="black", lw=3.0, alpha=0.55, zorder=7)
    ax.plot(cols, rows, "-", color="#ff2d55", lw=1.8, zorder=8)


def _endpoints(ax, start, goal):
    ax.plot(start[1], start[0], "o", mfc="#2b6cff", mec="white", mew=1.2, ms=11, zorder=9)
    ax.plot(goal[1], goal[0], "o", mfc="#ff3b30", mec="white", mew=1.2, ms=11, zorder=9)
    ax.annotate("S", (start[1], start[0]), xytext=(5, 5), textcoords="offset points", fontsize=12,
                fontweight="bold", color="white", zorder=10,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="black")])
    ax.annotate("G", (goal[1], goal[0]), xytext=(5, 5), textcoords="offset points", fontsize=12,
                fontweight="bold", color="white", zorder=10,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="black")])


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, ax.get_images()[0].get_extent()[1] - 0.0)
    ax.set_ylim(ax.get_images()[0].get_extent()[2] + 0.0, -0.5)


def figure_search_overlay(out_path, ctx: dict) -> None:
    """Figure A: field + trajectory + sources + expanded nodes + final path, per weighting."""
    grid, occ, start, goal = ctx["grid"], ctx["occupancy"], ctx["start"], ctx["goal"]
    weights = [w for w in ctx["weights"]]
    fig, axes = plt.subplots(1, len(weights), figsize=(7.4 * len(weights), 7.2), dpi=110)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(BG)
    image = None
    for ax, weight in zip(axes, weights):
        res = ctx["results"][(weight, "B3_linear_fusion", 0.5)]
        image = _field(ax, occ, ctx["fields"][weight], weight)
        pos = ctx["sources"]
        ax.scatter(pos[:, 1], pos[:, 0], c=ctx["progress"], cmap="plasma", vmin=0, vmax=1,
                   s=9, edgecolors="white", linewidths=0.3, zorder=6)
        _trajectory(ax, ctx["raw"])
        _expansion(ax, grid, res["trace"], grid.width)
        _path(ax, grid, res["path"])
        _endpoints(ax, start, goal)
        _clean(ax)
        ax.set_title(
            f"{weight} field  |  B3 linear fusion (alpha=0.5)\n"
            f"expanded={res['expanded']}  generated={res['generated']}  "
            f"len={res['length']}  L*={ctx['L_star']}  ratio={(res['length'] or 0) / ctx['L_star']:.2f}",
            fontsize=9,
        )
    if image is not None:
        cb = fig.colorbar(image, ax=axes.tolist(), fraction=0.03, pad=0.01)
        cb.set_label("guidance field F(x) (clipped)", fontsize=9)
    fig.suptitle(
        f"{ctx['scene_id']}  ·  {ctx['map_type']}  ·  {ctx['grade']}   |   field + M0 sources + "
        f"expansion (colour = expansion order) + final path",
        fontsize=11.5, fontweight="bold", y=0.995,
    )
    handles = [
        Line2D([], [], color=COL_FREE, lw=2, label="raw trajectory"),
        Line2D([], [], color="#ff2d55", lw=2, label="search path"),
        Line2D([], [], marker="s", ls="", mfc="orange", mec="none", label="expanded node"),
        Line2D([], [], marker="o", ls="", mfc="#2b6cff", mec="white", label="S"),
        Line2D([], [], marker="o", ls="", mfc="#ff3b30", mec="white", label="G"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def figure_expansion_compare(out_path, ctx: dict, weight: str, alpha: float) -> None:
    """Figure B: expansion behaviour of B0-B5 side by side on one map and field."""
    grid, occ, start, goal = ctx["grid"], ctx["occupancy"], ctx["start"], ctx["goal"]
    order = [
        ("B0_astar", None),
        ("B1_distance_gbfs", None),
        ("B2_field_only", None),
        ("B3_linear_fusion", alpha),
        ("B4_nonlinear_fusion", alpha),
        ("B5_adaptive_fusion", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 11.0), dpi=105)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_2d(axes)
    for ax, (method, a) in zip(axes.ravel(), order):
        res = ctx["results"].get((weight, method, a)) or ctx["results"].get(("none", method, None))
        if res is None:
            ax.axis("off")
            continue
        if method in ("B0_astar", "B1_distance_gbfs"):
            _background(ax, occ)
        else:
            _field(ax, occ, ctx["fields"][weight], weight)
        _trajectory(ax, ctx["raw"])
        _expansion(ax, grid, res["trace"], grid.width)
        _path(ax, grid, res["path"])
        _endpoints(ax, start, goal)
        _clean(ax)
        ax.set_title(
            f"{METHOD_LABELS[method]}" + (f"  (alpha={a})" if a is not None else "") + "\n"
            f"expanded={res['expanded']}  generated={res['generated']}  len={res['length']}  "
            f"ratio={(res['length'] or 0) / ctx['L_star']:.2f}",
            fontsize=9,
        )
    fig.suptitle(
        f"{ctx['scene_id']}  ·  {ctx['map_type']}  ·  {ctx['grade']}   |   expansion comparison "
        f"on the {weight} field   (colour = expansion order, L*={ctx['L_star']})",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure C: metric summaries
# --------------------------------------------------------------------------- #
def _method_series(records, method, field, key):
    grades = ("feasible", "mild_collision", "moderate_collision", "severe_collision")
    out = []
    for g in grades:
        vals = [r[key] for r in records
                if r["method"] == method and r["field"] == field and r["grade"] == g and r.get(key) is not None]
        vals = [v for v in vals if v is not None]
        out.append(float(np.mean(vals)) if vals else np.nan)
    return out


def summary_figures(outdir, records, by_grade, by_method) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    grades = ("feasible", "mild_collision", "moderate_collision", "severe_collision")
    xlabels = [g.replace("_collision", "") for g in grades]
    colors = {
        "B0_astar": "#333333",
        "B1_distance_gbfs": "#1f77b4",
        "B2_field_only": "#2ca02c",
        "B3_linear_fusion": "#ff7f0e",
        "B4_nonlinear_fusion": "#d62728",
        "B5_adaptive_fusion": "#9467bd",
    }

    # 1) expanded nodes per grade, averaged over alpha, linear + quadratic
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4), dpi=110)
    for ax, key, label in zip(axes, ("expanded", "search_time", "optimality_ratio"),
                              ("expanded nodes", "search time [s]", "path length / L*")):
        for method in colors:
            ys = []
            for g in grades:
                vals = [r[key] for r in records if r["method"] == method and r["grade"] == g
                        and r.get(key) is not None]
                ys.append(float(np.mean(vals)) if vals else np.nan)
            ax.plot(range(4), ys, "o-", color=colors[method], label=METHOD_LABELS[method], lw=1.8, ms=4)
        ax.set_xticks(range(4))
        ax.set_xticklabels(xlabels, fontsize=8.5)
        ax.set_title(f"{label} by trajectory grade", fontsize=9.5)
        ax.grid(alpha=0.3)
        if key == "optimality_ratio":
            ax.axhline(1.0, color="gray", ls=":", lw=1)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Search behaviour vs. guidance-trajectory collision grade (all 20 maps, both fields)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(outdir / "expanded_by_grade.png", facecolor=BG)
    plt.close(fig)

    # 2) efficiency / quality trade-off
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=110)
    ax = axes[0]
    for method in colors:
        for field, marker in (("linear", "o"), ("quadratic", "s")):
            xs, ys = [], []
            for r in records:
                if r["method"] == method and r["field"] == field and r.get("optimality_ratio") is not None:
                    xs.append(r["optimality_ratio"])
                    ys.append(r["expanded"])
            if xs:
                ax.scatter(np.mean(xs), np.mean(ys), color=colors[method], marker=marker, s=70,
                           edgecolors="white", linewidths=0.6,
                           label=f"{METHOD_LABELS[method]} · {field}" if method != "B0_astar" else "A*")
    ax.set_xlabel("mean path length / L*")
    ax.set_ylabel("mean expanded nodes")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.set_title("expansion vs. path quality (mean per (method, field))", fontsize=9.5)
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes[1]
    labels, search_t, build_t = [], [], []
    for method in colors:
        for field in ("linear", "quadratic"):
            if method in ("B0_astar", "B1_distance_gbfs") and field == "quadratic":
                continue
            sub = [r for r in records if r["method"] == method and r["field"] == field]
            if not sub:
                continue
            labels.append(f"{method.split('_')[0]}\n{field[:3]}")
            search_t.append(float(np.mean([r["search_time"] for r in sub])) * 1000)
            build_t.append(float(np.mean([r["field_build_time"] for r in sub])) * 1000)
    ax.bar(range(len(labels)), search_t, color="#4c72b0", label="search")
    ax.bar(range(len(labels)), build_t, bottom=search_t, color="#dd8452", label="field build")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("mean time [ms]")
    ax.set_title("search vs. field-build cost (total = stack)", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Cost structure: search time alone hides the field construction cost", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outdir / "cost_and_tradeoff.png", facecolor=BG)
    plt.close(fig)

    # 3) alpha / gamma sweep
    fusion = [r for r in records if r["method"] in ("B3_linear_fusion", "B4_nonlinear_fusion")]
    if fusion:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), dpi=110)
        alphas = sorted({r["alpha"] for r in fusion if r["alpha"] is not None})
        for ax, key, label in zip(axes, ("expanded", "optimality_ratio"), ("expanded nodes", "length / L*")):
            for method, ls in (("B3_linear_fusion", "-"), ("B4_nonlinear_fusion", "--")):
                for field, marker in (("linear", "o"), ("quadratic", "s")):
                    ys = []
                    for a in alphas:
                        vals = [r[key] for r in fusion
                                if r["method"] == method and r["field"] == field and r["alpha"] == a
                                and r.get(key) is not None]
                        ys.append(float(np.mean(vals)) if vals else np.nan)
                    ax.plot(alphas, ys, ls, marker=marker, color=colors[method],
                            label=f"{METHOD_LABELS[method]} · {field}", lw=1.6, ms=5)
            ax.set_xlabel("alpha (weight of the distance term)")
            ax.set_title(f"{label} vs. alpha", fontsize=9.5)
            ax.grid(alpha=0.3)
            if key == "optimality_ratio":
                ax.axhline(1.0, color="gray", ls=":", lw=1)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Fusion-parameter sweep (gamma={0.5} for B4)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(outdir / "alpha_sweep.png", facecolor=BG)
        plt.close(fig)
