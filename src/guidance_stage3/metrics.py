"""Aggregation helpers for the search-stage results."""

from __future__ import annotations

import numpy as np

METRIC_KEYS = (
    "expanded",
    "generated",
    "search_time",
    "field_build_time",
    "total_time",
    "path_length",
    "optimality_ratio",
    "mean_field_on_expanded",
    "mean_distance_on_expanded",
    "success",
)


def summarize(records: list[dict], group_keys: tuple[str, ...]) -> dict:
    """mean/std/min/max of every metric per group (missing values skipped)."""
    groups: dict[tuple, list[dict]] = {}
    for rec in records:
        groups.setdefault(tuple(rec[k] for k in group_keys), []).append(rec)
    out = {}
    for key, items in groups.items():
        bucket = {"n": len(items)}
        for metric in METRIC_KEYS:
            vals = np.array([float(r[metric]) for r in items if r.get(metric) is not None])
            vals = vals[np.isfinite(vals)]
            if vals.size:
                bucket[metric] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
        bucket["success_rate"] = float(np.mean([1.0 if r["status"] == "success" else 0.0 for r in items]))
        out["|".join(str(k) for k in key)] = bucket
    return out


def compare_to_baseline(records: list[dict]) -> dict:
    """Per (scene, grade, method) expansion ratio against B0, so the headline is explicit."""
    base = {}
    for rec in records:
        if rec["method"] == "B0_astar":
            base[(rec["scene_id"], rec["grade"])] = rec
    out = {}
    for rec in records:
        b = base.get((rec["scene_id"], rec["grade"]))
        if not b or rec["expanded"] == 0:
            continue
        key = f"{rec['method']}|{rec['field']}|a{rec.get('alpha')}|g{rec.get('gamma')}"
        out.setdefault(key, []).append(
            {
                "expanded_vs_astar": rec["expanded"] / b["expanded"],
                "length_vs_astar": (rec["path_length"] or 0) / b["path_length"] if b["path_length"] else None,
            }
        )
    summary = {}
    for key, items in out.items():
        exp = np.array([i["expanded_vs_astar"] for i in items])
        ln = np.array([i["length_vs_astar"] for i in items if i["length_vs_astar"] is not None])
        summary[key] = {
            "n": len(items),
            "expanded_over_astar_mean": float(exp.mean()),
            "expanded_over_astar_median": float(np.median(exp)),
            "length_over_astar_mean": float(ln.mean()) if ln.size else None,
        }
    return summary
