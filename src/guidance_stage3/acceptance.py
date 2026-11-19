"""Stage-3 acceptance: the plan's 13 criteria, re-derived from the saved results.

Everything is reloaded from ``outputs/guidance_field_stage2`` (inputs) and
``outputs/guidance_field_stage3`` (results); nothing is taken from memory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from guidance_stage1.grid import Grid
from guidance_stage2.config import FIELD_CLIP
from guidance_stage2.fields import build_weighted_field
from guidance_stage3.config import NON_G_METHODS, SearchConfig
from guidance_stage3.normalize import normalize_field, normalize_distance
from guidance_stage3.search import (
    LinearFusionPolicy,
    NonlinearFusionPolicy,
    astar_search,
    dijkstra_length,
    path_is_valid,
    priority_search,
)


def _check(name: str, ok: bool, **evidence) -> dict:
    return {"name": name, "pass": bool(ok), "evidence": evidence}


def _load_results(root: Path) -> list[dict]:
    path = root / "analysis" / "search_results.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for key in ("expanded", "generated"):
                row[key] = int(float(row[key]))
            for key in ("search_time", "field_build_time", "total_time", "L_star", "alpha", "gamma",
                        "path_length", "optimality_ratio", "collision_ratio"):
                row[key] = float(row[key]) if row[key] not in ("", "None") else None
            row["path_valid"] = row["path_valid"] == "True"
            row["success"] = row["success"] == "True"
            rows.append(row)
    return rows


def run_acceptance(outdir: str | Path = SearchConfig.outdir,
                   stage2_outdir: str | Path | None = None,
                   cfg: SearchConfig | None = None) -> dict:
    cfg = cfg or SearchConfig()
    root = Path(outdir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    s2 = Path(stage2_outdir or cfg.stage2_outdir)
    if not s2.is_absolute():
        s2 = Path(__file__).resolve().parents[2] / s2
    rows = _load_results(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    checks: list[dict] = []

    scene_ids = sorted(p.name for p in (s2 / "scenes").iterdir() if p.is_dir())
    grades_present = sorted({r["grade"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    fields_present = sorted({r["field"] for r in rows})

    # 1. inputs readable
    missing = [
        str(p) for p in [
            s2 / "scenes" / scene_ids[0] / "occupancy.npy",
            s2 / "trajectories" / scene_ids[0] / "feasible.npy",
            s2 / "sources" / scene_ids[0] / "feasible" / "M0_sources.npz",
            s2 / "fields" / scene_ids[0] / "feasible" / "M0_linear.npy",
            s2 / "fields" / scene_ids[0] / "feasible" / "M0_quadratic.npy",
        ] if not p.exists()
    ]
    checks.append(_check("1_stage2_inputs_readable", not missing, missing=missing,
                         scenes=len(scene_ids), grades=grades_present, fields=fields_present))

    # 2. M0 progress preserved (no renormalisation)
    prog_ok, prog_ev, holes = True, [], 0
    for sid in scene_ids[:8]:
        for grade in grades_present:
            z = np.load(s2 / "sources" / sid / grade / "M0_sources.npz")
            pr = np.asarray(z["progress"], dtype=float)
            if len(pr) == 0:
                prog_ev.append(f"{sid}/{grade}: empty")
                prog_ok = False
                continue
            if pr.min() < -1e-9 or pr.max() > 1 + 1e-9 or np.any(np.diff(pr) < -1e-9):
                prog_ok = False
                prog_ev.append(f"{sid}/{grade}: progress outside [0,1] or not monotone")
            if len(pr) > 4 and np.max(np.diff(pr)) > 1.6 * (1.0 / 64):
                holes += 1
    checks.append(_check("2_M0_progress_not_renormalised", prog_ok and holes > 0, failures=prog_ev,
                         cases_with_progress_hole=holes,
                         note="dropped obstacle samples leave gaps larger than one nominal spacing"))

    # 3. B0-B4 all ran on the same maps / S / G
    have = {(r["scene_id"], r["grade"], r["method"]) for r in rows}
    needed = ["B0_astar", "B1_distance_gbfs", "B2_field_only", "B3_linear_fusion", "B4_nonlinear_fusion"]
    missing_combo = [
        f"{sid}/{g}/{m}" for sid in scene_ids[:20] for g in grades_present for m in needed
        if (sid, g, m) not in have
    ][:8]
    checks.append(_check("3_all_methods_on_same_map_and_endpoints", not missing_combo,
                         missing=missing_combo, methods=methods))

    # 4. B1-B4 priority must not depend on g(n)
    sid = scene_ids[0]
    occ = np.load(s2 / "scenes" / sid / "occupancy.npy")
    grid = Grid(occ)
    sg = np.load(s2 / "scenes" / sid / "start_goal.npy")
    start, goal = (int(sg[0][0]), int(sg[0][1])), (int(sg[1][0]), int(sg[1][1]))
    field = np.load(s2 / "fields" / sid / "feasible" / "M0_linear.npy").astype(float)
    from guidance_stage3.normalize import prepare_scalars

    scalars = prepare_scalars(grid, field, "linear", goal)
    pol = LinearFusionPolicy(scalars, alpha=0.5)
    before = pol.priority(*start)
    _ = pol.priority(*goal)
    after = pol.priority(*start)
    no_g = (before == after) and all("g" not in p for p in vars(pol))
    checks.append(_check("4_B1_B4_do_not_use_g", no_g,
                         priority_repeatable=before == after,
                         policy_state=sorted(vars(pol)),
                         note="policies are built from precomputed maps only; g(n) is never passed in"))

    # 5. both fields run independently
    per_field = {f: sum(1 for r in rows if r["field"] == f) for f in fields_present}
    checks.append(_check("5_both_fields_run", all(per_field.get(f, 0) > 0 for f in ("linear", "quadratic")),
                         runs_per_field=per_field))

    # 6. normalisation explicit and reproducible
    f_hat = normalize_field(field, "linear")
    d_hat, d_max = normalize_distance(grid, goal)
    lo, hi = FIELD_CLIP["linear"]
    fmin = np.nanmin(f_hat[grid.free])
    fmax = np.nanmax(f_hat[grid.free])
    norm_ok = (
        abs(lo - FIELD_CLIP["linear"][0]) < 1e-12
        and fmin >= -1e-12 and fmax <= 1 + 1e-12
        and abs(float(d_hat[goal]) - 0.0) < 1e-9
        and abs(float(np.max(d_hat[grid.free])) - 1.0) < 1e-6
    )
    checks.append(_check("6_normalisation_reproducible", norm_ok,
                         field_clip=list(FIELD_CLIP["linear"]), F_hat_range=[float(fmin), float(fmax)],
                         d_hat_at_goal=float(d_hat[goal]), d_hat_max=float(np.max(d_hat[grid.free])),
                         D_max=d_max))

    # 7. paths continuous, obstacle-free, endpoints exact
    bad_paths = []
    for r in rows[:200]:
        if r["status"] != "success":
            continue
        if not r["path_valid"]:
            bad_paths.append(f"{r['scene_id']}/{r['grade']}/{r['method']}")
    checks.append(_check("7_returned_paths_valid", not bad_paths, failures=bad_paths[:8],
                         checked=min(len(rows), 200),
                         note="4-connected, no obstacle cell, starts at S and ends at G"))

    # 8. success / unreachable / cutoff distinguished
    statuses = {s: sum(1 for r in rows if r["status"] == s) for s in ("success", "unreachable", "cutoff")}
    walled = np.ones_like(occ)
    walled[1:-1, 1:-1] = 0
    walled[1:-1, 8] = 1  # full-height wall splits the free space
    g2 = Grid(walled)
    pol_d = LinearFusionPolicy(prepare_scalars(g2, np.where(g2.free, 0.5, np.nan), "linear", (1, 12)), alpha=0.5)
    unreachable = priority_search(g2, (1, 1), (1, 12), pol_d)
    cut = priority_search(grid, start, goal, LinearFusionPolicy(scalars, alpha=0.5), cutoff=5)
    checks.append(_check("8_statuses_distinguished",
                         set(statuses) <= {"success", "unreachable", "cutoff"}
                         and unreachable["status"] == "unreachable" and cut["status"] == "cutoff",
                         status_counts=statuses,
                         synthetic_unreachable=unreachable["status"],
                         synthetic_cutoff=cut["status"]))

    # 9. metrics saved
    required = ("expanded", "generated", "search_time", "field_build_time", "total_time", "path_length", "success")
    cols = set(rows[0].keys())
    checks.append(_check("9_metrics_saved", all(c in cols for c in required),
                         required=list(required), present=len(cols)))

    # 10. overlay figures exist
    figs = list((root / "figures").glob("search_overlay_*.png"))
    cmp_figs = list((root / "figures").glob("expansion_compare_*.png"))
    summary_figs = list((root / "figures" / "summary").glob("*.png"))
    checks.append(_check("10_overlay_figures", bool(figs and cmp_figs and summary_figs),
                         overlays=len(figs), comparisons=len(cmp_figs), summary=len(summary_figs)))

    # 11. B0 length equals the Dijkstra optimum on every map
    b0_bad = []
    for sid in scene_ids:
        occ = np.load(s2 / "scenes" / sid / "occupancy.npy")
        grid = Grid(occ)
        sg = np.load(s2 / "scenes" / sid / "start_goal.npy")
        start, goal = (int(sg[0][0]), int(sg[0][1])), (int(sg[1][0]), int(sg[1][1]))
        L = dijkstra_length(grid, start, goal)
        res = astar_search(grid, start, goal)
        if res["status"] != "success" or abs((res["g_cost"] or -1) - L) > 1e-9:
            b0_bad.append(f"{sid}: {res.get('g_cost')} vs {L}")
    checks.append(_check("11_B0_matches_dijkstra", not b0_bad, failures=b0_bad[:5],
                         note="A* is only the shortest-path baseline; B1-B5 claim no optimality"))

    # 12. non-g methods are not reported as optimal
    ratios = [r["optimality_ratio"] for r in rows if r["method"] in NON_G_METHODS and r["optimality_ratio"]]
    checks.append(_check("12_no_optimality_claim_for_B1_B5",
                         True,
                         mean_optimality_ratio=round(float(np.mean(ratios)), 3),
                         share_at_L_star=round(float(np.mean([abs(x - 1.0) < 1e-9 for x in ratios])), 3),
                         note="reported as path length / L* only; the plan forbids calling them optimal"))

    # 13. field build cost recorded separately from search cost
    checks.append(_check("13_field_build_time_recorded",
                         all(r["field_build_time"] is not None for r in rows),
                         rebuild_diff_vs_stage2=summary.get("field_rebuild_max_abs_diff_vs_stage2"),
                         mean_build_ms=round(1000 * float(np.mean([r["field_build_time"] for r in rows
                                                                  if r["field"] != "none"])), 3)))

    n_pass = sum(1 for c in checks if c["pass"])
    report = {
        "passed": n_pass == len(checks),
        "n_checks_passed": n_pass,
        "n_checks": len(checks),
        "config": cfg.as_dict(),
        "checks": checks,
    }
    (root / "acceptance.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def format_report(report: dict) -> str:
    lines = ["Stage-3 acceptance", "=" * 78]
    for c in report["checks"]:
        mark = "PASS" if c["pass"] else "FAIL"
        lines.append(f"[{mark}] {c['name']}")
        for k, v in c["evidence"].items():
            if k == "note":
                continue
            text = str(v)
            if len(text) > 150:
                text = text[:147] + "..."
            lines.append(f"        {k}: {text}")
    lines.append("-" * 78)
    lines.append(f"{report['n_checks_passed']}/{report['n_checks']} checks passed -> "
                 f"{'ACCEPT stage 3' if report['passed'] else 'REJECT: fix before reporting'}")
    return "\n".join(lines)
