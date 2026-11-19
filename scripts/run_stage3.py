"""Run the stage-3 search experiment: guidance fields -> B0..B5 search policies.

    python scripts/run_stage3.py                 # 20 maps x 4 grades x 18 configurations
    python scripts/run_stage3.py --scenes 4      # first 4 maps (fast smoke run)
    python scripts/run_stage3.py --no-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_stage3.config import SearchConfig  # noqa: E402
from guidance_stage3.pipeline import run  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=SearchConfig.outdir)
    p.add_argument("--stage2-outdir", default=SearchConfig.stage2_outdir)
    p.add_argument("--scenes", type=int, default=None, help="limit to the first N maps")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    cfg = SearchConfig(outdir=args.outdir, stage2_outdir=args.stage2_outdir)
    summary = run(cfg, make_figures=not args.no_figures, scene_limit=args.scenes)
    t = summary["totals"]
    print(
        f"\nmaps={t['maps']} runs={t['runs']} success={t['success']} "
        f"valid_paths={t['paths_valid']} status={t['status_counts']} runtime={summary['runtime_s']}s"
    )
    print(f"field rebuild vs stage 2: max abs diff = {summary['field_rebuild_max_abs_diff_vs_stage2']:.2e}")
    print(f"outputs -> {Path(cfg.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
