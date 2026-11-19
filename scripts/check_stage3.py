"""Audit the stage-3 search results against the plan's 13 acceptance criteria.

    python scripts/check_stage3.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_stage3.acceptance import format_report, run_acceptance  # noqa: E402
from guidance_stage3.config import SearchConfig  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=SearchConfig.outdir)
    p.add_argument("--stage2-outdir", default=SearchConfig.stage2_outdir)
    args = p.parse_args(argv)
    report = run_acceptance(args.outdir, args.stage2_outdir, SearchConfig(outdir=args.outdir,
                                                                         stage2_outdir=args.stage2_outdir))
    text = format_report(report)
    print(text)
    out = Path(args.outdir)
    if not out.is_absolute():
        out = ROOT / out
    (out / "acceptance.txt").write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
