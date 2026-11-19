"""Contract tests for the core package (stdlib unittest, no pytest needed)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guidance_field import (
    Grid,
    GuidanceField,
    astar,
    build_field,
    build_guidance_field,
    guided_search,
    path_is_valid,
    propagate_heap,
    propagate_scipy,
    resample_polyline,
    trajectory_sources,
)


def border_room(h=16, w=16, walls=()):
    occ = np.zeros((h, w), dtype=np.int8)
    occ[0, :] = occ[-1, :] = 1
    occ[:, 0] = occ[:, -1] = 1
    for r, c in walls:
        occ[r, c] = 1
    return occ


class GridTests(unittest.TestCase):
    def test_neighbours_respect_obstacles(self):
        grid = Grid(border_room(walls=[(5, c) for c in range(1, 10)]))
        inside = grid.idx(5, 1)
        for u in grid.neighbors(inside):
            self.assertTrue(grid.free.flat[u])

    def test_bfs_distance(self):
        grid = Grid(border_room())
        dist = grid.bfs_distance([grid.idx(2, 2)])
        self.assertEqual(dist[2, 2], 0)
        self.assertEqual(dist[2, 5], 3)
        self.assertEqual(dist[0, 0], -1)  # obstacle

    def test_unreachable_region_is_minus_one(self):
        occ = border_room(walls=[(r, 8) for r in range(1, 15)])
        grid = Grid(occ)
        dist = grid.bfs_distance([grid.idx(3, 3)])
        self.assertEqual(dist[3, 12], -1)


class SourceTests(unittest.TestCase):
    def test_resample_spans_the_polyline(self):
        pts = np.array([[2.0, 2.0], [2.0, 10.0]])
        points, progress = resample_polyline(pts, 9)
        self.assertEqual(len(points), 9)
        np.testing.assert_allclose(progress[[0, -1]], [0.0, 1.0])
        np.testing.assert_allclose(points[0], pts[0])
        np.testing.assert_allclose(points[-1], pts[-1])

    def test_drop_and_project_handle_obstacles(self):
        occ = border_room(walls=[(8, c) for c in range(2, 9)])
        grid = Grid(occ)
        traj = np.array([[8.0, 3.0], [8.0, 8.0]])          # straight through the wall
        dropped = trajectory_sources(grid, traj, num_samples=20, method="drop")
        projected = trajectory_sources(grid, traj, num_samples=20, method="project")
        self.assertGreater(dropped["num_dropped"], 0)
        self.assertEqual(projected["num_dropped"], 0)
        for src in (dropped, projected):
            pos = src["positions_rc"]
            self.assertTrue(np.all(grid.free[pos[:, 0], pos[:, 1]]))
            self.assertTrue(np.all(np.diff(src["progress"]) >= -1e-9))

    def test_dropping_keeps_a_progress_hole(self):
        occ = border_room(walls=[(8, c) for c in range(2, 9)])
        grid = Grid(occ)
        traj = np.array([[5.0, 5.0], [12.0, 5.0]])   # crosses the wall at row 8
        src = trajectory_sources(grid, traj, num_samples=20, method="drop")
        self.assertGreater(src["num_dropped"], 0)
        self.assertGreater(float(np.max(np.diff(src["progress"]))), 1.0 / 20)


class FieldTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(border_room(walls=[(8, c) for c in range(2, 12)]))
        self.traj = np.array([[3.0, 3.0], [3.0, 12.0], [12.0, 12.0]])
        self.src = trajectory_sources(self.grid, self.traj, num_samples=32)

    def test_obstacles_are_nan(self):
        field = build_field(self.grid, self.src["positions_rc"], self.src["progress"])
        self.assertTrue(np.all(np.isnan(field[~self.grid.free])))
        self.assertTrue(np.all(np.isfinite(field[self.grid.free])))

    def test_backends_agree(self):
        pos, u = self.src["positions_rc"], self.src["progress"]
        a = build_field(self.grid, pos, u, weight="linear", method="heap")
        b = build_field(self.grid, pos, u, weight="linear", method="scipy")
        ok = self.grid.free & np.isfinite(a) & np.isfinite(b)
        self.assertLess(float(np.max(np.abs(a[ok] - b[ok]))), 1e-9)

    def test_field_decays_with_distance_and_respects_weights(self):
        pos, u = self.src["positions_rc"], self.src["progress"]
        uniform = build_field(self.grid, pos, u, weight="uniform")
        self.assertAlmostEqual(float(np.nanmax(uniform)), 1.0, places=9)
        linear = build_field(self.grid, pos, u, weight="linear")
        # the start of the trajectory must not outrank its end
        self.assertLess(float(linear[3, 3]), float(linear[12, 12]))

    def test_duplicate_sources_keep_the_strongest(self):
        field = build_field(self.grid, np.array([[5, 5], [5, 5]]), np.array([0.0, 1.0]),
                            weight="linear")
        self.assertAlmostEqual(float(field[5, 5]), 1.0, places=9)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.occ = border_room(h=20, w=20, walls=[(8, c) for c in range(1, 16)])
        self.grid = Grid(self.occ)
        self.start, self.goal = (3, 3), (16, 16)

    def test_astar_is_optimal_and_valid(self):
        res = astar(self.grid, self.start, self.goal)
        self.assertEqual(res["status"], "success")
        info = path_is_valid(self.grid, res["path"], self.start, self.goal)
        self.assertTrue(info["valid"], msg=info)
        best = int(self.grid.bfs_distance([self.grid.idx(*self.start)])[self.goal])
        self.assertEqual(res["length"], best)

    def test_guided_search_beats_astar_on_expansions(self):
        traj = np.array([[3.0, 3.0], [15.0, 3.0], [15.0, 16.0], [16.0, 16.0]])
        gfield = GuidanceField(self.occ)
        gfield.build(traj, num_samples=48)
        base = astar(self.grid, self.start, self.goal)
        guided = gfield.search(self.start, self.goal)
        self.assertEqual(guided["status"], "success")
        self.assertTrue(path_is_valid(self.grid, guided["path"], self.start, self.goal)["valid"])
        self.assertLess(guided["expanded"], base["expanded"])

    def test_alpha_one_is_pure_distance_greedy(self):
        traj = np.array([[3.0, 3.0], [16.0, 16.0]])
        gfield = GuidanceField(self.occ)
        gfield.build(traj)
        res = gfield.search(self.start, self.goal, alpha=1.0)
        self.assertEqual(res["status"], "success")

    def test_unreachable_goal_is_reported(self):
        occ = border_room(h=20, w=20)
        occ[1:19, 9] = 1  # full-height wall splits the map
        grid = Grid(occ)
        self.assertFalse(grid.free[16, 16] is None)
        res = astar(grid, (3, 3), (16, 16))
        self.assertEqual(res["status"], "unreachable")
        self.assertEqual(len(res["path"]), 0)

    def test_cutoff_is_distinct_from_unreachable(self):
        res = astar(self.grid, self.start, self.goal, cutoff=5)
        self.assertEqual(res["status"], "cutoff")


class ApiTests(unittest.TestCase):
    def test_one_shot_functions_match_the_class(self):
        occ = border_room(h=20, w=20, walls=[(8, c) for c in range(1, 16)])
        traj = np.array([[3.0, 3.0], [3.0, 16.0], [16.0, 16.0]])
        grid = Grid(occ)
        field = build_guidance_field(occ, traj, num_samples=32)
        res = guided_search(occ, (3, 3), (16, 16), field=field)
        self.assertEqual(res["status"], "success")
        self.assertTrue(path_is_valid(grid, res["path"], (3, 3), (16, 16))["valid"])

    def test_search_before_build_raises(self):
        gfield = GuidanceField(border_room())
        with self.assertRaises(RuntimeError):
            gfield.search((3, 3), (10, 10))

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            GuidanceField(border_room(), weight="cubic")
        with self.assertRaises(ValueError):
            GuidanceField(border_room(), source_method="bridge")
        with self.assertRaises(ValueError):
            trajectory_sources(Grid(border_room()), np.zeros((3, 3)))

    def test_field_is_reusable_across_searches(self):
        occ = border_room(h=20, w=20, walls=[(8, c) for c in range(1, 16)])
        traj = np.array([[3.0, 3.0], [16.0, 16.0]])
        gfield = GuidanceField(occ)
        field = gfield.build(traj)
        np.testing.assert_array_equal(field, gfield.field)
        a = gfield.search((3, 3), (16, 16), alpha=0.0)
        b = gfield.search((3, 3), (16, 16), alpha=0.5)
        self.assertEqual(a["status"], "success")
        self.assertEqual(b["status"], "success")


if __name__ == "__main__":
    unittest.main()
