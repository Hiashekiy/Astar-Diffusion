"""Run the stage-1 guidance-field experiment.

    python scripts/run_stage1.py                 # full run: 20 maps, 4 grades, 3 fields
    python scripts/run_stage1.py --quick          # 1 map per scene type (smoke test)
    python scripts/run_stage1.py --no-figures     # data only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_stage1.config import Stage1Config  # noqa: E402
from guidance_stage1.pipeline import run  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=Stage1Config.outdir)
    parser.add_argument("--seed", type=int, default=Stage1Config.seed)
    parser.add_argument("--instances", type=int, default=Stage1Config.instances_per_type)
    parser.add_argument("--samples", type=int, default=Stage1Config.num_samples)
    parser.add_argument("--lam", type=float, default=Stage1Config.lam)
    parser.add_argument("--quick", action="store_true", help="one instance per scene type")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-lambda-sweep", action="store_true")
    args = parser.parse_args(argv)

    cfg = Stage1Config(
        outdir=args.outdir,
        seed=args.seed,
        instances_per_type=1 if args.quick else args.instances,
        num_samples=args.samples,
        lam=args.lam,
    )
    summary = run(cfg, make_figures=not args.no_figures, lambda_sweep=not args.no_lambda_sweep)
    totals = summary["totals"]
    print(
        f"\nmaps={totals['maps']} trajectories={totals['trajectories']} fields={totals['fields']} "
        f"all-grades={totals['maps_with_all_grades']}  runtime={summary['runtime_s']}s"
    )
    print(f"outputs -> {Path(cfg.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
