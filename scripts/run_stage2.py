"""Run the stage-2A experiment: colliding trajectories -> guidance fields.

    python scripts/run_stage2.py            # 20 maps x 4 grades x 3 methods x 3 weights = 720 fields
    python scripts/run_stage2.py --quick     # 1 map per family
    python scripts/run_stage2.py --no-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_stage2.config import Stage2Config  # noqa: E402
from guidance_stage2.pipeline import run  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=Stage2Config.outdir)
    parser.add_argument("--seed", type=int, default=Stage2Config.seed)
    parser.add_argument("--instances", type=int, default=Stage2Config.instances_per_type)
    parser.add_argument("--samples", type=int, default=Stage2Config.num_source_samples)
    parser.add_argument("--lam", type=float, default=Stage2Config.lam)
    parser.add_argument("--quick", action="store_true", help="one instance per map family")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    cfg = Stage2Config(
        outdir=args.outdir,
        seed=args.seed,
        instances_per_type=1 if args.quick else args.instances,
        num_source_samples=args.samples,
        lam=args.lam,
    )
    summary = run(cfg, make_figures=not args.no_figures)
    totals = summary["totals"]
    print(
        f"\nmaps={totals['maps']} trajectories={totals['trajectories']} fields={totals['field_cases']} "
        f"maps_with_all_grades={totals['maps_with_all_grades']}  runtime={summary['runtime_s']}s"
    )
    if summary["missing_collision_grades"]:
        print("missing collision grades:")
        for item in summary["missing_collision_grades"]:
            for miss in item["missing_grades"]:
                print(f"  {item['scene_id']} {item['map_type']}: {miss['grade']} "
                      f"(band {miss['band']}, best available {miss['best_available']:.3f})")
    print(f"outputs -> {Path(cfg.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
