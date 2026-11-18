"""Audit the saved stage-1 outputs against the seven acceptance criteria.

    python scripts/check_acceptance.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_stage1.acceptance import format_report, run_acceptance  # noqa: E402
from guidance_stage1.config import Stage1Config  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=Stage1Config.outdir)
    parser.add_argument("--lam", type=float, default=Stage1Config.lam)
    args = parser.parse_args(argv)
    report = run_acceptance(args.outdir, Stage1Config(outdir=args.outdir, lam=args.lam))
    text = format_report(report)
    print(text)
    out = Path(args.outdir)
    if not out.is_absolute():
        out = ROOT / out
    (out / "acceptance.txt").write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
