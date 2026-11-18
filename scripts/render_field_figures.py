"""Render one full-size image per guidance field, straight from the saved artefacts.

One file = one (map, trajectory grade, source method, weighting) field, drawn on the whole
map with its trajectory, sources and colorbar.  Nothing is recomputed: occupancy,
trajectories, sources and fields are reloaded from ``outputs/``.

    python scripts/render_field_figures.py                        # linear weights, all 20 maps (240 images)
    python scripts/render_field_figures.py --weights linear,uniform,quadratic   # 720 images
    python scripts/render_field_figures.py --scenes scene_001 --grades severe_collision
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_stage2.config import SOURCE_METHODS, TRAJECTORY_GRADES, WEIGHT_KINDS, Stage2Config  # noqa: E402
from guidance_stage2.visualization import figure_single_field  # noqa: E402


def _load_metrics(root: Path) -> dict:
    path = root / "analysis" / "per_case_metrics.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["scene_id"], row["grade"], row["method"], row["weight"])] = row
    return out


def _fmt(value, pattern: str = "{:+.3f}") -> str:
    if value in (None, "", "None", "nan"):
        return "--"
    try:
        return pattern.format(float(value))
    except (TypeError, ValueError):
        return "--"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=Stage2Config.outdir)
    parser.add_argument("--weights", default="linear", help="comma separated: linear,uniform,quadratic")
    parser.add_argument("--methods", default=",".join(SOURCE_METHODS))
    parser.add_argument("--grades", default="", help="comma separated grades (default: all present)")
    parser.add_argument("--scenes", default="", help="comma separated scene ids (default: all)")
    args = parser.parse_args(argv)

    root = Path(args.outdir)
    if not root.is_absolute():
        root = ROOT / root
    weights = [w.strip() for w in args.weights.split(",") if w.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    only_grades = {g.strip() for g in args.grades.split(",") if g.strip()}
    only_scenes = {s.strip() for s in args.scenes.split(",") if s.strip()}
    metrics = _load_metrics(root)

    scenes = sorted(p.name for p in (root / "scenes").iterdir() if p.is_dir())
    if only_scenes:
        scenes = [s for s in scenes if s in only_scenes]

    written = 0
    for sid in scenes:
        occ = np.load(root / "scenes" / sid / "occupancy.npy")
        sg = np.load(root / "scenes" / sid / "start_goal.npy")
        meta = json.loads((root / "scenes" / sid / "metadata.json").read_text(encoding="utf-8"))
        traj_meta = json.loads((root / "trajectories" / sid / "metadata.json").read_text(encoding="utf-8"))
        start = (int(sg[0][0]), int(sg[0][1]))
        goal = (int(sg[1][0]), int(sg[1][1]))
        grades = [g for g in TRAJECTORY_GRADES if g in traj_meta["trajectories"]]
        if only_grades:
            grades = [g for g in grades if g in only_grades]
        out_dir = root / "figures" / sid / "fields"
        out_dir.mkdir(parents=True, exist_ok=True)

        for grade in grades:
            entry = traj_meta["trajectories"][grade]
            raw = np.load(root / "trajectories" / sid / f"{grade}.npy")
            intervals = entry["collision_intervals"]
            for method in methods:
                npz = np.load(root / "sources" / sid / grade / f"{method}_sources.npz")
                sources = npz["positions_rc"]
                progress = npz["progress"]
                support = npz["support_path"] if "support_path" in npz else None
                for weight in weights:
                    field_file = root / "fields" / sid / grade / f"{method}_{weight}.npy"
                    if not field_file.exists():
                        continue
                    field = np.load(field_file).astype(np.float64)
                    row = metrics.get((sid, grade, method, weight), {})
                    title = (f"{sid}  ·  {meta['type']} {meta['type_name']}  ·  {grade}  ·  "
                             f"{method} + {weight} weights")
                    subtitle = (
                        f"collision_ratio={_fmt(entry['metrics']['collision_ratio'], '{:.3f}')}    "
                        f"collision intervals={entry['metrics']['collision_interval_count']}    "
                        f"sources={len(sources)}/64    ΔF={_fmt(row.get('delta_F'), '{:+.3f}')}    "
                        f"MAE vs reference field={_fmt(row.get('mae_all_free'), '{:.3f}')}"
                    )
                    figure_single_field(
                        out_dir / f"{grade}__{method}__{weight}.png",
                        occ, field, title, subtitle, weight,
                        raw=raw, intervals=intervals,
                        support=support if method == "M2" else None,
                        sources=sources, source_progress=progress,
                        start=start, goal=goal,
                        caption=(f"global guidance field F(x) = max_i [ w(u_i) − λ·d_free(x, s_i) ]   "
                                 f"λ={Stage2Config.lam}   M={Stage2Config.num_source_samples}   "
                                 f"map {meta['size']}×{meta['size']}   seed {meta['seed']}"),
                    )
                    written += 1
        print(f"[{sid}] fields rendered", flush=True)

    print(f"\n{written} field figures -> {root / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
