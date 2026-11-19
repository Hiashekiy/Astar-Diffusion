"""Stage-2A unit tests (stdlib unittest): collision geometry, M0/M1/M2, fields.

Covers the twelve cases required by the plan plus the grading rule.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_stage1.fields import weight_at_progress  # noqa: E402
from guidance_stage1.grid import Grid  # noqa: E402
from guidance_stage2.collision import (  # noqa: E402
    detect_collision_intervals,
    measure_collision_ratio,
    project_to_nearest_free,
    segment_collision_intervals,
    trace_segment_cells,
)
from guidance_stage2.config import Stage2Config  # noqa: E402
from guidance_stage2.fields import build_weighted_field, source_weights  # noqa: E402
from guidance_stage2.scenes import generate_scene  # noqa: E402
from guidance_stage2.source_methods import (  # noqa: E402
    build_all_methods,
    build_local_bridge,
    build_sources_bridge,
    build_sources_drop,
    build_sources_project,
    sample_raw_trajectory,
)
from guidance_stage2.trajectories import generate_trajectories, select_by_collision_band  # noqa: E402

CFG = Stage2Config()


def grid_with(walls, size=10):
    occ = np.zeros((size, size), dtype=np.int8)
    for r, c in walls:
        occ[r, c] = 1
    return Grid(occ), occ


WALL_COL = [(r, 4) for r in range(0, 8)]  # vertical wall col 4, rows 0..7


class CollisionTests(unittest.TestCase):
    def test_1_segment_with_free_endpoints_still_collides(self):
        g, _ = grid_with(WALL_COL)
        self.assertTrue(g.free[4, 1] and g.free[4, 7])
        ivs = segment_collision_intervals(g, (4.0, 1.0), (4.0, 7.0))
        self.assertEqual(len(ivs), 1)
        self.assertGreater(ivs[0][1] - ivs[0][0], 0.0)

    def test_2_free_segment_has_zero_collision(self):
        g, _ = grid_with(WALL_COL)
        self.assertEqual(segment_collision_intervals(g, (4.0, 5.0), (4.0, 7.0)), [])
        info = measure_collision_ratio(g, np.array([[8.0, 0.0], [8.0, 9.0]]))
        self.assertAlmostEqual(info["collision_length"], 0.0)
        self.assertAlmostEqual(info["collision_ratio"], 0.0)

    def test_3_single_cell_crossing_interval_is_correct(self):
        g, _ = grid_with(WALL_COL)
        # straight through the wall cell (4, 4): 1 cell of a 6-long segment
        info = detect_collision_intervals(g, np.array([[4.0, 1.0], [4.0, 7.0]]))
        self.assertEqual(info["collision_interval_count"], 1)
        self.assertAlmostEqual(info["collision_length"], 1.0, places=6)
        self.assertAlmostEqual(info["collision_ratio"], 1.0 / 6.0, places=6)
        iv = info["intervals"][0]
        # the wall cell (4,4) occupies columns 3.5..4.5, i.e. arc 2.5..3.5 of the length-6 segment
        self.assertAlmostEqual(iv["u_in"], 2.5 / 6.0, places=6)
        self.assertAlmostEqual(iv["u_out"], 3.5 / 6.0, places=6)

    def test_4_two_separated_crossings_give_two_intervals(self):
        walls = [(4, c) for c in range(0, 10)] + [(7, c) for c in range(0, 10)]
        g, _ = grid_with(walls, size=12)
        info = detect_collision_intervals(g, np.array([[1.0, 4.0], [10.0, 4.0]]))
        self.assertEqual(info["collision_interval_count"], 2)
        self.assertAlmostEqual(info["collision_length"], 2.0, places=6)

    def test_trace_segment_covers_all_crossed_cells(self):
        g, _ = grid_with(WALL_COL)
        cells = trace_segment_cells((4.0, 1.0), (4.0, 7.0))
        cols = [c for _r, c, _a, _b in cells]
        self.assertEqual(cols, [1, 2, 3, 4, 5, 6, 7])
        for _r, _c, t_in, t_out in cells:
            self.assertLessEqual(t_in, t_out + 1e-12)

    def test_projection_respects_radius(self):
        g, _ = grid_with(WALL_COL)
        hit = project_to_nearest_free(g, (4, 4), 2.0)
        self.assertIsNotNone(hit)
        (r, c), dist = hit
        self.assertTrue(g.free[r, c])
        self.assertLessEqual(dist, 2.0)
        self.assertIsNone(project_to_nearest_free(g, (4, 4), 0.5))


class SourceMethodTests(unittest.TestCase):
    def setUp(self):
        self.g, self.occ = grid_with(WALL_COL)
        self.raw = np.array([[4.0, 1.0], [4.0, 7.0]])  # crosses (4,4)
        self.raw = np.array([[4.0, 1.0], [4.0, 2.0], [4.0, 3.0], [4.0, 4.0], [4.0, 5.0], [4.0, 6.0], [4.0, 7.0]])

    def test_5_M0_drops_obstacle_sources_and_keeps_progress(self):
        src = build_sources_drop(self.g, self.raw, 12, CFG)
        pos = src["positions_rc"]
        self.assertTrue(all(self.g.free[r, c] for r, c in pos))
        nominal = sample_raw_trajectory(self.raw, 12)["progress"]
        # every surviving progress value is one of the original sample progresses
        for u in src["progress"]:
            self.assertTrue(np.min(np.abs(nominal - u)) < 1e-9)
        # the dropped samples are exactly the ones whose cell is an obstacle
        nominal_cells = sample_raw_trajectory(self.raw, 12)["cells"]
        n_drop = sum(1 for r, c in nominal_cells if not self.g.free[r, c])
        self.assertEqual(len(pos), 12 - n_drop)
        self.assertGreater(n_drop, 0)

    def test_6_M1_projects_into_free_space_within_radius(self):
        src = build_sources_project(self.g, self.raw, 16, CFG)
        self.assertTrue(all(self.g.free[r, c] for r, c in src["positions_rc"]))
        md = src["metadata"]
        self.assertIsNotNone(md["max_projection_distance"])
        self.assertLessEqual(md["max_projection_distance"], CFG.projection_radius + 1e-9)
        self.assertEqual(len(src["positions_rc"]), 16)

    def test_7_M2_bridge_is_free_and_local(self):
        bridge = build_local_bridge(self.g, (4, 3), (4, 5), (1, 2, 4), CFG)
        self.assertIsNotNone(bridge)
        pts = bridge["polyline"]
        for r, c in pts:
            self.assertTrue(self.g.free[int(r), int(c)], msg=f"bridge point {(r, c)} is an obstacle")
        info = measure_collision_ratio(self.g, pts)
        self.assertAlmostEqual(info["collision_ratio"], 0.0)
        # the straight distance is 2, the bridge has to go around the wall -> longer
        self.assertGreater(bridge["length"], bridge["straight_distance"])

    def test_8_M2_bridge_progress_increases_from_entry_to_exit(self):
        src = build_sources_bridge(self.g, self.raw, 32, CFG)
        self.assertEqual(src["metadata"]["num_bridged_intervals"], 1)
        self.assertTrue(all(self.g.free[r, c] for r, c in src["positions_rc"]))
        pr = src["progress"]
        self.assertTrue(np.all(np.diff(pr) >= -1e-12))
        origin = src["origin"]
        self.assertIn("local_bridge", set(origin))
        bridged = src["positions_rc"][origin == "local_bridge"]
        self.assertGreater(len(bridged), 0)

    def test_9_all_methods_weight_from_their_own_stored_progress(self):
        for builder in (build_sources_drop, build_sources_project, build_sources_bridge):
            src = builder(self.g, self.raw, 24, CFG)
            pr = src["progress"]
            for kind in ("uniform", "linear", "quadratic"):
                expected = np.asarray(weight_at_progress(pr, kind), dtype=float)
                got = source_weights(pr, kind)
                np.testing.assert_allclose(got, expected)
                if kind == "linear":
                    np.testing.assert_allclose(got, pr)  # NOT re-normalised by index


class PlotCoordinateTests(unittest.TestCase):
    """Regression: everything is stored as (row, col) but matplotlib draws (x, y) = (col, row)."""

    asym = np.array([[2.0, 5.0], [8.0, 5.0], [8.0, 12.0]])

    def test_colored_segments_keep_row_col_order(self):
        from guidance_stage2.visualization import _colored_segments

        segs, colors = _colored_segments(self.asym, [])
        self.assertEqual(len(segs), len(self.asym) - 1)
        np.testing.assert_allclose(segs[0][0], [2.0, 5.0])  # still (row, col)
        np.testing.assert_allclose(segs[-1][-1], [8.0, 12.0])

    def test_draw_trajectory_emits_col_row_coordinates(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from guidance_stage2.visualization import draw_trajectory

        fig, ax = plt.subplots()
        draw_trajectory(ax, self.asym, [])
        line = ax.collections[-1]
        segs = line.get_segments()
        first, last = segs[0], segs[-1]
        np.testing.assert_allclose(first[0], [5.0, 2.0])  # (x, y) = (col, row)
        np.testing.assert_allclose(last[-1], [12.0, 8.0])
        plt.close(fig)

    def test_drawn_endpoints_match_start_and_goal_markers(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from guidance_stage2.visualization import _mark, draw_trajectory

        start, goal = (2, 5), (8, 12)
        raw = np.array([[2.0, 5.0], [8.0, 5.0], [8.0, 12.0]])
        fig, ax = plt.subplots()
        draw_trajectory(ax, raw, [])
        segs = ax.collections[-1].get_segments()
        np.testing.assert_allclose(segs[0][0], [start[1], start[0]])
        np.testing.assert_allclose(segs[-1][-1], [goal[1], goal[0]])
        _mark(ax, start, "#2b6cff")
        marker = ax.lines[-1]
        np.testing.assert_allclose([marker.get_xdata()[0], marker.get_ydata()[0]], [start[1], start[0]])
        plt.close(fig)

    def test_collision_and_free_colours_still_split_correctly(self):
        from guidance_stage2.visualization import COL_COLLIDE, COL_FREE, _colored_segments

        raw = np.array([[2.0, 5.0], [8.0, 5.0], [8.0, 12.0]])
        intervals = [{"s_in": 1.0, "s_out": 2.0}]  # middle of the second segment
        segs, colors = _colored_segments(raw, intervals)
        self.assertIn(COL_FREE, colors)
        self.assertIn(COL_COLLIDE, colors)
        for seg in segs:
            self.assertEqual(np.asarray(seg).shape, (2, 2))


class FieldTests(unittest.TestCase):
    def test_10_propagation_matches_closed_form(self):
        g, _ = grid_with(WALL_COL)
        src = build_sources_bridge(g, np.array([[4.0, 1.0], [4.0, 7.0]]), 8, CFG)
        kind = "linear"
        result = build_weighted_field(g, src, kind, CFG.lam)
        pos, pr = src["positions_rc"], src["progress"]
        w = source_weights(pr, kind)
        d_all = np.stack([g.distance_map([g.idx(int(r), int(c))]) for r, c in pos])
        model = np.max(w[:, None, None] - CFG.lam * d_all, axis=0)
        ok = g.free & np.isfinite(result["field"])
        self.assertLess(float(np.max(np.abs(result["field"][ok] - model[ok]))), 1e-9)

    def test_11_obstacles_stay_nan(self):
        g, _ = grid_with(WALL_COL)
        src = build_sources_bridge(g, np.array([[4.0, 1.0], [4.0, 7.0]]), 8, CFG)
        for kind in ("uniform", "linear", "quadratic"):
            field = build_weighted_field(g, src, kind, CFG.lam)["field"]
            self.assertTrue(np.all(np.isnan(field[~g.free])))
            self.assertTrue(np.all(np.isfinite(field[g.free])))


class ReproducibilityTests(unittest.TestCase):
    def test_12_same_seed_same_map_and_trajectories(self):
        a = generate_scene(0, "B", CFG.seed + 5000, CFG)
        b = generate_scene(0, "B", CFG.seed + 5000, CFG)
        np.testing.assert_array_equal(a["occupancy"], b["occupancy"])
        self.assertEqual(a["metadata"]["start"], b["metadata"]["start"])
        self.assertEqual(a["metadata"]["goal"], b["metadata"]["goal"])

        grid = Grid(a["occupancy"])
        s, t = tuple(a["metadata"]["start"]), tuple(a["metadata"]["goal"])
        tr1 = generate_trajectories(grid, a["occupancy"], s, t, CFG, np.random.default_rng(3))
        tr2 = generate_trajectories(grid, a["occupancy"], s, t, CFG, np.random.default_rng(3))
        for grade in tr1["trajectories"]:
            np.testing.assert_array_equal(
                tr1["trajectories"][grade]["raw_path"], tr2["trajectories"][grade]["raw_path"]
            )
            self.assertAlmostEqual(
                tr1["trajectories"][grade]["metrics"]["collision_ratio"],
                tr2["trajectories"][grade]["metrics"]["collision_ratio"],
            )


class GradingTests(unittest.TestCase):
    def test_grades_follow_measured_ratio_not_generator_name(self):
        cands = [
            {"key": 1, "generator": "shortcut", "metrics": {"collision_ratio": 0.030, "path_length": 10.0}},
            {"key": 2, "generator": "bezier", "metrics": {"collision_ratio": 0.100, "path_length": 11.0}},
            {"key": 3, "generator": "directed_crossing", "metrics": {"collision_ratio": 0.220, "path_length": 12.0}},
            {"key": 4, "generator": "bezier", "metrics": {"collision_ratio": 0.600, "path_length": 13.0}},
        ]
        chosen, missing = select_by_collision_band(cands, CFG.collision_bands())
        self.assertEqual(chosen["mild_collision"]["key"], 1)
        self.assertEqual(chosen["moderate_collision"]["key"], 2)
        self.assertEqual(chosen["severe_collision"]["key"], 3)
        self.assertEqual(missing, [])

    def test_unreachable_band_is_reported_and_not_faked(self):
        cands = [{"key": 1, "generator": "bezier", "metrics": {"collision_ratio": 0.02, "path_length": 10.0}}]
        chosen, missing = select_by_collision_band(cands, CFG.collision_bands())
        self.assertIn("mild_collision", chosen)
        self.assertNotIn("severe_collision", chosen)
        self.assertEqual([m["grade"] for m in missing], ["moderate_collision", "severe_collision"])

    def test_all_methods_share_progress_definition(self):
        g, _ = grid_with(WALL_COL)
        raw = np.array([[4.0, 1.0], [4.0, 3.0], [4.0, 4.0], [4.0, 7.0]])
        sources = build_all_methods(g, raw, CFG)
        for method, src in sources.items():
            self.assertTrue(np.all(src["progress"] >= 0.0))
            self.assertTrue(np.all(src["progress"] <= 1.0))
            self.assertTrue(all(g.free[r, c] for r, c in src["positions_rc"]), msg=method)


if __name__ == "__main__":
    unittest.main()
