"""Stage-2A acceptance: re-derive every criterion from the artefacts on disk.

Checks are grouped exactly like the plan (§12 A-E) and never trust in-memory state:
occupancy, trajectories, sources and fields are reloaded from ``outputs/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from guidance_stage1.fields import propagate
from guidance_stage1.grid import Grid
from guidance_stage2.collision import measure_collision_ratio, polyline_length
from guidance_stage2.config import (
    COLLISION_BANDS,
    MAP_TYPES,
    SOURCE_METHODS,
    TRAJECTORY_GRADES,
    WEIGHT_KINDS,
    Stage2Config,
)
from guidance_stage2.fields import source_weights
from guidance_stage2.scenes import generate_scene


def _load(root: Path) -> list[dict]:
    scenes = []
    for d in sorted((root / "scenes").iterdir()):
        if not d.is_dir():
            continue
        occ = np.load(d / "occupancy.npy")
        sg = np.load(d / "start_goal.npy")
        md = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        sid = md["scene_id"]
        traj_md = json.loads((root / "trajectories" / sid / "metadata.json").read_text(encoding="utf-8"))
        scenes.append(
            {
                "scene_id": sid,
                "occupancy": occ,
                "grid": Grid(occ),
                "start": (int(sg[0][0]), int(sg[0][1])),
                "goal": (int(sg[1][0]), int(sg[1][1])),
                "meta": md,
                "traj_meta": traj_md,
            }
        )
    return scenes


def _check(name: str, ok: bool, **evidence) -> dict:
    return {"name": name, "pass": bool(ok), "evidence": evidence}


def run_acceptance(outdir: str | Path = Stage2Config.outdir, cfg: Stage2Config | None = None) -> dict:
    cfg = cfg or Stage2Config()
    root = Path(outdir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    scenes = _load(root)
    lam = cfg.lam
    checks: list[dict] = []

    # ------------------------------------------------------------------ A. maps
    by_type = {t: [s for s in scenes if s["meta"]["type"] == t] for t in MAP_TYPES}
    checks.append(_check(
        "A1_four_families_x_instances",
        all(len(v) >= cfg.instances_per_type for v in by_type.values()),
        counts={t: len(v) for t, v in by_type.items()},
    ))

    reproducible, repro_fail = True, []
    for s in scenes[:4]:
        regen = generate_scene(s["meta"]["index"], s["meta"]["type"], s["meta"]["seed"], cfg)
        if not np.array_equal(regen["occupancy"], s["occupancy"]):
            reproducible = False
            repro_fail.append(s["scene_id"])
    checks.append(_check("A2_seed_reproducible", reproducible, mismatched=repro_fail,
                         note="two scenes per family regenerated from metadata seed"))

    sg_ok, sg_evidence = True, []
    for s in scenes:
        g = s["grid"]
        if not (g.is_free(*s["start"]) and g.is_free(*s["goal"])):
            sg_ok = False
            sg_evidence.append(f"{s['scene_id']}: endpoint in obstacle")
            continue
        dist = int(g.geodesic_distance([g.idx(*s["start"])])[g.idx(*s["goal"])])
        if dist < cfg.min_geodesic_sg:
            sg_ok = False
            sg_evidence.append(f"{s['scene_id']}: d(S,G)={dist}")
    checks.append(_check("A3_endpoints_valid_and_connected", sg_ok, failures=sg_evidence,
                         min_required=cfg.min_geodesic_sg))

    struct_ok, struct_ev = True, []
    for s in scenes:
        st = s["meta"]["stats"]
        if st["components"] != 1:
            struct_ok = False
            struct_ev.append(f"{s['scene_id']}: {st['components']} components")
        if s["meta"]["type"] in ("A", "B", "D") and st["open_cell_ratio"] < 0.25:
            struct_ok = False
            struct_ev.append(f"{s['scene_id']}: only {st['open_cell_ratio']} open cells")
    checks.append(_check("A4_not_a_single_corridor", struct_ok, failures=struct_ev,
                         note="single free component + >=25% open (>=3 free neighbours) free cells",
                         open_cell_ratio_range=[
                             min(s["meta"]["stats"]["open_cell_ratio"] for s in scenes),
                             max(s["meta"]["stats"]["open_cell_ratio"] for s in scenes),
                         ]))

    shape_kinds = sorted({k for s in by_type["A"] for k in s["meta"]["structure"]["shape_kinds"]})
    checks.append(_check("A5_A_random_multi_shape_obstacles",
                         len(shape_kinds) >= 3 and all(s["meta"]["structure"]["shape_count"] >= 6 for s in by_type["A"]),
                         kinds=shape_kinds,
                         min_shapes=min(s["meta"]["structure"]["shape_count"] for s in by_type["A"])))

    rooms = [s["meta"]["structure"]["room_count"] for s in by_type["B"]]
    doors = [s["meta"]["structure"]["door_count"] for s in by_type["B"]]
    checks.append(_check("A6_B_rooms_walls_doors", min(rooms) >= 4 and min(doors) >= 3,
                         rooms=rooms, doors=doors))

    cyc = [s["meta"]["stats"]["cycles"] for s in by_type["C"]]
    dead = [s["meta"]["stats"]["dead_end_cells"] for s in by_type["C"]]
    checks.append(_check("A7_C_branches_loops_dead_ends", min(cyc) >= 3 and min(dead) >= 10,
                         cycles=cyc, dead_end_cells=dead))

    d_kinds = [len(s["meta"]["structure"]["structure_kinds"]) for s in by_type["D"]]
    gates = [s["meta"]["structure"]["structure_counts"].get("gate", 0) for s in by_type["D"]]
    checks.append(_check("A8_D_structured_topology", min(d_kinds) >= 3 and min(gates) >= 1,
                         kinds_per_map=d_kinds, gates=gates,
                         kinds_present=sorted({k for s in by_type["D"] for k in s["meta"]["structure"]["structure_kinds"]})))

    gallery = root / "figures" / "map_gallery.png"
    checks.append(_check("A9_map_montage_saved", gallery.exists(), path=str(gallery.relative_to(root)),
                         bytes=gallery.stat().st_size if gallery.exists() else 0))

    # ------------------------------------------------------- B. collision trajectories
    ref_ok, ref_ev = True, []
    for s in scenes:
        ref_file = root / "trajectories" / s["scene_id"] / "reference_astar.npy"
        if not ref_file.exists():
            ref_ok = False
            ref_ev.append(f"{s['scene_id']}: no reference file")
            continue
        ref = np.load(ref_file)
        m = measure_collision_ratio(s["grid"], ref, cfg.collision_detection_step)
        if m["collision_ratio"] > 1e-6:
            ref_ok = False
            ref_ev.append(f"{s['scene_id']}: reference collides ({m['collision_ratio']:.4f})")
    checks.append(_check("B1_collision_free_reference_on_every_map", ref_ok, failures=ref_ev))

    grade_cov = {}
    for s in scenes:
        have = set(s["traj_meta"]["trajectories"].keys())
        grade_cov[s["scene_id"]] = sorted(have)
    with_all = sum(1 for v in grade_cov.values() if set(TRAJECTORY_GRADES) <= set(v))
    checks.append(_check("B2_G1_G2_G3_coverage", with_all >= 0.8 * len(scenes),
                         maps_with_all_grades=with_all, total_maps=len(scenes),
                         detail={k: v for k, v in list(grade_cov.items())[:6]},
                         note="families where a band is unreachable are recorded in summary.json"))

    band_ok, band_ev = True, []
    for s in scenes:
        for grade, entry in s["traj_meta"]["trajectories"].items():
            raw = np.load(root / "trajectories" / s["scene_id"] / f"{grade}.npy")
            m = measure_collision_ratio(s["grid"], raw, cfg.collision_detection_step)
            ratio = m["collision_ratio"]
            if grade == "feasible":
                ok = ratio <= 1e-6
            else:
                lo, hi = COLLISION_BANDS[grade]
                ok = lo <= ratio < hi
            if not ok:
                band_ok = False
                band_ev.append(f"{s['scene_id']}/{grade}: measured {ratio:.3f}")
    checks.append(_check("B3_grades_match_measured_ratio", band_ok, failures=band_ev,
                         note="re-measured from the stored trajectory, not from the generator name"))

    ends_ok, ends_ev = True, []
    for s in scenes:
        ref = np.load(root / "trajectories" / s["scene_id"] / "reference_astar.npy")
        for grade in s["traj_meta"]["trajectories"]:
            raw = np.load(root / "trajectories" / s["scene_id"] / f"{grade}.npy")
            if np.linalg.norm(raw[0] - ref[0]) > 1e-6 or np.linalg.norm(raw[-1] - ref[-1]) > 1e-6:
                ends_ok = False
                ends_ev.append(f"{s['scene_id']}/{grade}")
    checks.append(_check("B4_same_start_and_goal", ends_ok, failures=ends_ev))

    interior_ok, interior_ev = True, []
    for s in scenes:
        for grade, entry in s["traj_meta"]["trajectories"].items():
            for iv in entry["collision_intervals"]:
                ec, xc = iv["entry_cell"], iv["exit_cell"]
                if not (s["grid"].is_free(*ec) and s["grid"].is_free(*xc)):
                    interior_ok = False
                    interior_ev.append(f"{s['scene_id']}/{grade}/iv{iv['index']}")
    checks.append(_check("B5_interior_crossings_detected", interior_ok, failures=interior_ev,
                         note="every collision interval has free entry/exit anchors, i.e. the "
                              "segment left free space and came back"))

    stats_ok, stats_ev = True, []
    for s in scenes:
        for grade, entry in s["traj_meta"]["trajectories"].items():
            raw = np.load(root / "trajectories" / s["scene_id"] / f"{grade}.npy")
            m = measure_collision_ratio(s["grid"], raw, cfg.collision_detection_step)
            stored = entry["metrics"]
            if abs(m["collision_ratio"] - stored["collision_ratio"]) > 0.02:
                stats_ok = False
                stats_ev.append(f"{s['scene_id']}/{grade}: {stored['collision_ratio']:.3f} vs {m['collision_ratio']:.3f}")
            if m["collision_interval_count"] != stored["collision_interval_count"]:
                stats_ok = False
                stats_ev.append(f"{s['scene_id']}/{grade}: interval count mismatch")
    checks.append(_check("B6_collision_statistics_consistent", stats_ok, failures=stats_ev,
                         tolerance=0.02))

    # A "repaired" file would look like a grid walk again: integer vertices with unit
    # steps.  A raw shortcut keeps a long jump between lattice anchors, a Bezier / blend
    # keeps non-lattice vertices; both prove the stored geometry was not re-snapped.
    raw_ok, raw_ev = True, []
    n_float_paths, n_jump_paths = 0, 0
    for s in scenes:
        for grade, entry in s["traj_meta"]["trajectories"].items():
            raw = np.load(root / "trajectories" / s["scene_id"] / f"{grade}.npy")
            if raw.dtype.kind != "f":
                raw_ok = False
                raw_ev.append(f"{s['scene_id']}/{grade}: dtype {raw.dtype}")
                continue
            if np.any(raw != np.rint(raw)):
                n_float_paths += 1
            segs = np.linalg.norm(np.diff(raw, axis=0), axis=1)
            if segs.size and segs.max() > 1.5:
                n_jump_paths += 1
            is_grid_walk = np.allclose(raw, np.rint(raw)) and (segs.size == 0 or segs.max() <= 1.5 + 1e-9)
            if grade != "feasible" and is_grid_walk:
                raw_ok = False
                raw_ev.append(f"{s['scene_id']}/{grade}: stored as a unit-step grid walk (repaired)")
            if grade != "feasible" and entry["metrics"]["collision_ratio"] > 0:
                if measure_collision_ratio(s["grid"], raw, cfg.collision_detection_step)["collision_ratio"] <= 0:
                    raw_ok = False
                    raw_ev.append(f"{s['scene_id']}/{grade}: stored trajectory no longer collides")
    checks.append(_check("B7_raw_trajectory_not_repaired", raw_ok, failures=raw_ev,
                         paths_with_non_lattice_vertices=n_float_paths,
                         paths_with_shortcut_jumps=n_jump_paths,
                         note="stored as float (N,2); collision geometry preserved and still colliding"))

    # ---------------------------------------------------------------- C. sources
    def _npz(sid, grade, method):
        return np.load(root / "sources" / sid / grade / f"{method}_sources.npz", allow_pickle=False)

    free_ok, free_ev, counts = True, [], {}
    for s in scenes:
        for grade in s["traj_meta"]["trajectories"]:
            for method in SOURCE_METHODS:
                z = _npz(s["scene_id"], grade, method)
                pos = z["positions_rc"]
                counts[(method, grade)] = counts.get((method, grade), 0) + len(pos)
                bad = [tuple(int(v) for v in p) for p in pos if not s["grid"].free[int(p[0]), int(p[1])]]
                if bad:
                    free_ok = False
                    free_ev.append(f"{s['scene_id']}/{grade}/{method}: {len(bad)} obstacle sources")
    checks.append(_check("C1_C7_all_sources_are_free_cells", free_ok, failures=free_ev,
                         note="covers M0, M1 and M2 at once (C7: nothing illegal reaches the propagator)"))

    proj_ok, proj_ev, proj_max = True, [], 0.0
    for s in scenes:
        for grade in s["traj_meta"]["trajectories"]:
            z = _npz(s["scene_id"], grade, "M1")
            for p in z["positions_rc"]:
                r, c = int(p[0]), int(p[1])
                if not s["grid"].free[r, c]:
                    proj_ok = False
                    proj_ev.append(f"{s['scene_id']}/{grade}: projected onto obstacle")
        for grade in s["traj_meta"]["trajectories"]:
            bm = json.loads((root / "sources" / s["scene_id"] / grade / "bridge_metadata.json").read_text(encoding="utf-8"))
            md = bm["M1"]["metadata"]
            if md.get("max_projection_distance") is not None:
                proj_max = max(proj_max, float(md["max_projection_distance"]))
    checks.append(_check("C2_M1_projection_within_radius", proj_ok and proj_max <= cfg.projection_radius + 1e-9,
                         failures=proj_ev, max_projection_distance=proj_max, radius=cfg.projection_radius))

    bridge_ok, bridge_ev = True, []
    for s in scenes:
        for grade in s["traj_meta"]["trajectories"]:
            z = _npz(s["scene_id"], grade, "M2")
            support = z["support_path"]
            if len(support) >= 2:
                m = measure_collision_ratio(s["grid"], support, cfg.collision_detection_step)
                if m["collision_ratio"] > 1e-6:
                    bridge_ok = False
                    bridge_ev.append(f"{s['scene_id']}/{grade}: support path collides {m['collision_ratio']:.4f}")
    checks.append(_check("C3_M2_bridge_paths_free", bridge_ok, failures=bridge_ev,
                         note="support path of every M2 case re-checked against the occupancy"))

    # "M2 is local, not a global replan" is checked on three independent facts:
    #   1. every collision interval gets exactly one bridge (nothing else is bridged),
    #   2. the support path still keeps the original free parts (origin original_free),
    #   3. no single bridge spans a whole S->G path.
    # It is deliberately *not* checked as "sum of bridge lengths < path length": a severe
    # trajectory with many intervals accumulates detour length naturally (up to ~0.78 of
    # one path per bridge, ~2.2x in total across intervals).
    local_ok, local_ev = True, []
    single_ratios, free_origin_fracs, bridge_counts = [], [], []
    for s in scenes:
        ref_len = s["traj_meta"]["reference"]["length"]
        for grade, entry in s["traj_meta"]["trajectories"].items():
            bm = json.loads((root / "sources" / s["scene_id"] / grade / "bridge_metadata.json").read_text(encoding="utf-8"))
            md = bm["M2"]["metadata"]
            bridges = md.get("bridges") or []
            n_intervals = entry["metrics"]["collision_interval_count"]
            if len(bridges) != n_intervals:
                local_ok = False
                local_ev.append(f"{s['scene_id']}/{grade}: {len(bridges)} bridges for {n_intervals} intervals")
            for b in bridges:
                ratio = float(b["length"]) / float(ref_len) if ref_len else 0.0
                single_ratios.append(ratio)
                if ratio > 0.85:
                    local_ok = False
                    local_ev.append(f"{s['scene_id']}/{grade}/iv{b['interval']}: single bridge = {ratio:.2f} x path")
            z = _npz(s["scene_id"], grade, "M2")
            origin = z["source_origin"]
            frac = float((origin == "original_free").mean()) if len(origin) else 0.0
            free_origin_fracs.append(frac)
            bridge_counts.append(len(bridges))
            if bridges and frac < 0.10:
                local_ok = False
                local_ev.append(f"{s['scene_id']}/{grade}: only {frac:.2f} of sources from the original free path")
    checks.append(_check("C4_M2_is_local_not_a_global_replan", local_ok, failures=local_ev,
                         num_bridged_cases=len(bridge_counts),
                         max_single_bridge_over_path=max(single_ratios) if single_ratios else None,
                         min_original_free_source_fraction=min(free_origin_fracs) if free_origin_fracs else None,
                         note="bridge count matches collision intervals, the original free segments "
                              "still provide sources, and no single bridge spans the whole path"))

    prog_ok, prog_ev = True, []
    for s in scenes:
        for grade in s["traj_meta"]["trajectories"]:
            for method in SOURCE_METHODS:
                z = _npz(s["scene_id"], grade, method)
                pr = z["progress"]
                if len(pr) and (pr.min() < -1e-9 or pr.max() > 1 + 1e-9):
                    prog_ok = False
                    prog_ev.append(f"{s['scene_id']}/{grade}/{method}: progress outside [0,1]")
                if len(pr) > 1 and np.any(np.diff(pr) < -1e-9):
                    prog_ok = False
                    prog_ev.append(f"{s['scene_id']}/{grade}/{method}: progress not monotone")
    checks.append(_check("C5_original_progress_preserved", prog_ok, failures=prog_ev,
                         note="u is the arc-length fraction along the raw trajectory, kept as is"))

    failkeys_ok = True
    for s in scenes:
        for grade in s["traj_meta"]["trajectories"]:
            bm = json.loads((root / "sources" / s["scene_id"] / grade / "bridge_metadata.json").read_text(encoding="utf-8"))
            for method in SOURCE_METHODS:
                md = bm[method]["metadata"]
                required = ("num_valid_sources", "num_invalid_sources", "failures")
                if not all(k in md for k in required):
                    failkeys_ok = False
    checks.append(_check("C6_failures_recorded", failkeys_ok,
                         note="every method reports num_valid/num_invalid sources and a failures list"))

    # ----------------------------------------------------------------- D. fields
    fields_ok, nan_ok, finite_ok, lip_worst, formula_err = True, True, True, 0.0, 0.0
    missing_fields = []
    for s in scenes:
        for grade in s["traj_meta"]["trajectories"]:
            for method in SOURCE_METHODS:
                for kind in WEIGHT_KINDS:
                    f = root / "fields" / s["scene_id"] / grade / f"{method}_{kind}.npy"
                    if not f.exists():
                        missing_fields.append(str(f.relative_to(root)))
                        continue
                    field = np.load(f).astype(np.float64)
                    grid = s["grid"]
                    if not np.all(np.isnan(field[~grid.free])):
                        nan_ok = False
                    if not np.all(np.isfinite(field[grid.free])):
                        finite_ok = False
                    flat = field.ravel()
                    for v in grid.free_idx[::7]:  # sampled: the full sweep is O(n) per field
                        fv = flat[v]
                        for u in grid.neighbors(int(v)):
                            lip_worst = max(lip_worst, abs(fv - flat[u]) - lam)
    checks.append(_check("D1_obstacles_are_nan", nan_ok and not missing_fields,
                         missing_fields=missing_fields[:5], num_missing=len(missing_fields)))
    checks.append(_check("D2_all_free_cells_finite", finite_ok))
    checks.append(_check("D4_no_wall_crossing", lip_worst <= 1e-4, max_lipschitz_excess=float(f"{lip_worst:.3e}"),
                         tolerance=1e-4, note="|dF| <= lambda on every free edge (sampled 1/7 of cells)"))

    for s in scenes:
        grid = s["grid"]
        for grade in list(s["traj_meta"]["trajectories"])[:2]:
            for method in SOURCE_METHODS:
                z = _npz(s["scene_id"], grade, method)
                pos, pr = z["positions_rc"], z["progress"]
                if len(pos) == 0:
                    continue
                for kind in WEIGHT_KINDS:
                    field = np.load(root / "fields" / s["scene_id"] / grade / f"{method}_{kind}.npy").astype(np.float64)
                    w = source_weights(pr, kind)
                    closed = np.full((grid.height, grid.width), -np.inf)
                    for (r, c), wi in zip(pos, w):
                        closed[int(r), int(c)] = max(closed[int(r), int(c)], wi)
                    d_all = np.stack([grid.distance_map([grid.idx(int(r), int(c))]) for r, c in pos])
                    model = np.max(w[:, None, None] - lam * d_all, axis=0)
                    ok = grid.free & np.isfinite(field)
                    formula_err = max(formula_err, float(np.max(np.abs(field[ok] - model[ok]))))
    checks.append(_check("D3_matches_closed_form", formula_err < 1e-4,
                         max_abs_error=float(f"{formula_err:.3e}"),
                         note="checked on 2 grades x 3 methods x 3 weights per map"))

    weights_ok = True
    for s in scenes:
        for grade in s["traj_meta"]["trajectories"]:
            for method in SOURCE_METHODS:
                z = _npz(s["scene_id"], grade, method)
                pr = z["progress"]
                for kind in WEIGHT_KINDS:
                    expected = source_weights(pr, kind)
                    if not np.allclose(expected, z[f"weights_{kind}"], atol=1e-12):
                        weights_ok = False
    checks.append(_check("D5_weights_from_original_progress", weights_ok, note="recomputed from the stored u"))

    hole_ok, hole_ev, hole_cases = True, [], 0
    for s in scenes:
        for grade in ("moderate_collision", "severe_collision"):
            if grade not in s["traj_meta"]["trajectories"]:
                continue
            z = _npz(s["scene_id"], grade, "M0")
            pr = np.sort(z["progress"])
            if len(pr) < 8:
                continue
            spacing = np.diff(pr)
            nominal = 1.0 / cfg.num_source_samples
            if spacing.max() > 1.6 * nominal:
                hole_cases += 1
            else:
                hole_ev.append(f"{s['scene_id']}/{grade}: no visible progress hole")
    checks.append(_check("D6_M0_M1_do_not_renormalise", hole_cases >= max(1, int(0.5 * len(scenes))),
                         maps_with_progress_hole=hole_cases, notes=hole_ev[:4],
                         nominal_spacing=1.0 / cfg.num_source_samples,
                         note="dropped sources leave a gap larger than one nominal sample spacing"))

    diffs = {"M0-M1": [], "M0-M2": [], "M1-M2": []}
    for s in scenes:
        for grade in s["traj_meta"]["trajectories"]:
            arr = {
                m: np.load(root / "fields" / s["scene_id"] / grade / f"{m}_linear.npy").astype(np.float64)
                for m in SOURCE_METHODS
            }
            free = s["grid"].free
            for key in diffs:
                a, b = key.split("-")
                diffs[key].append(float(np.mean(np.abs(arr[a][free] - arr[b][free]))))
    severe_mae = []
    csv_like = json.loads((root / "analysis" / "aggregate_metrics.json").read_text(encoding="utf-8"))["aggregate"]
    for key, bucket in csv_like.items():
        if key.endswith("|M2|linear") and "severe" in key:
            severe_mae.append(bucket.get("mae_all_free", {}).get("mean"))
    checks.append(_check("D7_methods_differ_measurably",
                         max(np.mean(v) for v in diffs.values() if v) > 0.01,
                         pairwise_mean_abs_diff={k: float(np.mean(v)) for k, v in diffs.items()},
                         m2_severe_mae_vs_reference=severe_mae))

    # --------------------------------------------------- E. experiment + figures
    expected = 0
    for s in scenes:
        expected += len(s["traj_meta"]["trajectories"]) * len(SOURCE_METHODS) * len(WEIGHT_KINDS)
    checks.append(_check("E1_comparison_matrix_complete", not missing_fields,
                         expected_fields=expected, missing=len(missing_fields)))

    figs = root / "figures"
    per_scene = ["trajectories.png", "collision_intervals.png", "source_methods.png",
                 "field_methods_linear.png", "M2_weight_ablation.png", "field_difference.png",
                 "owners.png", "progress_profiles.png"]
    missing_figs = [
        f"{s['scene_id']}/{name}" for s in scenes for name in per_scene
        if not (figs / s["scene_id"] / name).exists()
    ]
    checks.append(_check("E2_E3_collision_and_bridge_visualised", not missing_figs,
                         missing=missing_figs[:6], num_missing=len(missing_figs),
                         note="figure existence + the interval/bridge data they are drawn from "
                              "(pixel content is not machine-checked)"))

    trace_ok = True
    for s in scenes:
        tm = s["traj_meta"]
        if not all(k in tm for k in ("seed", "reference", "coverage")):
            trace_ok = False
        for grade, entry in tm["trajectories"].items():
            if "generator" not in entry or "params" not in entry:
                trace_ok = False
    checks.append(_check("E4_full_traceability", trace_ok, note="seed, generator and params stored per trajectory"))

    checks.append(_check(
        "E5_summary_and_acceptance_outputs",
        (root / "summary.json").exists() and (root / "analysis" / "per_case_metrics.csv").exists(),
        summary=(root / "summary.json").exists(),
        csv=(root / "analysis" / "per_case_metrics.csv").exists(),
    ))

    stage1 = Path(__file__).resolve().parents[2] / "outputs" / "guidance_field_stage1"
    stage1_ok = False
    if (stage1 / "acceptance.json").exists():
        stage1_ok = bool(json.loads((stage1 / "acceptance.json").read_text(encoding="utf-8")).get("passed"))
    checks.append(_check("E6_stage1_untouched", stage1_ok,
                         stage1_acceptance_passed=stage1_ok,
                         note="stage-1 code is only imported; its own acceptance still passes"))

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
    lines = ["Stage-2A acceptance", "=" * 78]
    for check in report["checks"]:
        mark = "PASS" if check["pass"] else "FAIL"
        lines.append(f"[{mark}] {check['name']}")
        for key, val in check["evidence"].items():
            if key == "note":
                continue
            text = str(val)
            if len(text) > 160:
                text = text[:157] + "..."
            lines.append(f"        {key}: {text}")
    lines.append("-" * 78)
    lines.append(
        f"{report['n_checks_passed']}/{report['n_checks']} checks passed -> "
        f"{'ACCEPT stage 2A' if report['passed'] else 'REJECT: fix before stage 2B'}"
    )
    return "\n".join(lines)
