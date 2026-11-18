"""Audit the stage-2A artefacts against the plan's acceptance criteria (A-E).

    python scripts/check_stage2.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_stage2.acceptance import format_report, run_acceptance  # noqa: E402
from guidance_stage2.config import Stage2Config  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=Stage2Config.outdir)
    args = parser.parse_args(argv)
    report = run_acceptance(args.outdir, Stage2Config(outdir=args.outdir))
    text = format_report(report)
    print(text)
    out = Path(args.outdir)
    if not out.is_absolute():
        out = ROOT / out
    (out / "acceptance.txt").write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
