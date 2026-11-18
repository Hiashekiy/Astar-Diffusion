"""Unit tests for the stage-1 building blocks (stdlib unittest, no extra deps).

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_stage1.config import BUCKETS, Stage1Config  # noqa: E402
from guidance_stage1.fields import (  # noqa: E402
    lipschitz_violation,
    propagate,
    sample_progress,
    source_weights,
)
from guidance_stage1.grid import Grid, detour_profile  # noqa: E402
from guidance_stage1.paths import (  # noqa: E402
    astar,
    build_candidate_pool,
    concat,
    path_length,
    resample_arc_length,
    select_by_grade,
)
from guidance_stage1.scenes import generate_scene  # noqa: E402


def open_grid(h=16, w=16, walls=()):
    occ = np.zeros((h, w), dtype=np.int8)
    for r, c in walls:
        occ[r, c] = 1
    return Grid(occ)


class GridTests(unittest.TestCase):
    def test_geodesic_respects_walls(self):
        # a full-height wall with a single gap at the bottom row
        walls = [(r, 8) for r in range(0, 15)]
        g = open_grid(walls=walls)
        d = g.geodesic_distance([g.idx(0, 0)])
        self.assertEqual(int(d[g.idx(0, 0)]), 0)
        # straight across the wall is blocked: the only way round is via row 15
        self.assertEqual(int(d[g.idx(0, 10)]), 0 + 8 + 15 + 15 + 2)

    def test_cycle_rank_counts_independent_loops(self):
        self.assertEqual(open_grid(4, 4).cycle_rank(), 9)
        tree = np.zeros((5, 5), dtype=np.int8)
        tree[1:, :] = 1  # single row corridor: a path, no cycle
        tree[:, 2:] = 1
        tree[3, 0:3] = 0
        self.assertEqual(Grid(tree).cycle_rank(), 0)

    def test_detour_zero_on_optimal_route_positive_in_dead_end(self):
        occ = np.zeros((9, 9), dtype=np.int8)
        occ[1:8, 1:8] = 1
        occ[4, 1:8] = 0  # corridor
        occ[1:5, 6] = 0  # dead-end stub going up from the corridor
        g = Grid(occ)
        prof = detour_profile(g, (4, 1), (4, 7))
        self.assertAlmostEqual(float(prof["detour"][4, 4]), 0.0)
        # tip of the stub costs a full in-and-out detour
        self.assertAlmostEqual(float(prof["detour"][1, 6]), 6.0)

    def test_geodesic_ball_does_not_cross_walls(self):
        walls = [(r, 8) for r in range(0, 16)]
        g = open_grid(walls=walls)
        ball = g.geodesic_ball((5, 5), 3)
        self.assertTrue(ball[5, 5])
        self.assertFalse(ball[5, 9])  # across the wall, geodesically far


class PathTests(unittest.TestCase):
    def test_astar_is_optimal(self):
        rng = np.random.default_rng(3)
        for _ in range(5):
            occ = (rng.random((20, 20)) < 0.25).astype(np.int8)
            occ[0, 0] = occ[19, 19] = 0
            g = Grid(occ)
            s, t = g.idx(0, 0), g.idx(19, 19)
            path = astar(g, s, t)
            dist = g.geodesic_distance([s])
            if dist[t] < 0:
                self.assertIsNone(path)
                continue
            self.assertIsNotNone(path)
            self.assertEqual(path_length(path), int(dist[t]))
            self.assertEqual(path[0], s)
            self.assertEqual(path[-1], t)

    def test_weighted_astar_respects_suboptimality_bound(self):
        rng = np.random.default_rng(11)
        occ = (rng.random((24, 24)) < 0.3).astype(np.int8)
        occ[0, :] = 0  # keep one route open so A* always succeeds
        occ[:, 23] = 0
        g = Grid(occ)
        s, t = g.idx(0, 0), g.idx(23, 23)
        best = path_length(astar(g, s, t))
        for hw in (1.5, 2.0):
            longer = path_length(astar(g, s, t, h_weight=hw))
            self.assertLessEqual(longer, hw * best + 1e-9)

    def test_resample_spacing_and_endpoints(self):
        g = open_grid()
        nodes = [g.idx(0, c) for c in range(16)]  # straight line, L = 15
        samples = resample_arc_length(nodes, g.width, 8)
        self.assertEqual(tuple(samples[0]), (0, 0))
        self.assertEqual(tuple(samples[-1]), (0, 15))
        self.assertEqual(len({tuple(r) for r in samples}), 8)
        steps = np.diff(samples[:, 1])
        self.assertTrue(np.all(steps > 0))  # monotone along the arc

    def test_resample_allows_repeats_only_when_path_doubles_back(self):
        g = open_grid()
        out = [g.idx(5, c) for c in range(4, 12)]
        back = list(reversed(out[:-1]))
        nodes = out + back  # go right then come back over the same cells
        samples = resample_arc_length(nodes, g.width, 16)
        self.assertLess(len({tuple(r) for r in samples}), 16)

    def test_concat_drops_junction_duplicate(self):
        self.assertEqual(concat([1, 2, 3], [3, 4]), [1, 2, 3, 4])
        self.assertEqual(concat([1, 2], [7, 8]), [1, 2, 7, 8])
        self.assertEqual(concat([1, 2], []), [])


class FieldTests(unittest.TestCase):
    def setUp(self):
        self.a = np.zeros((4, 3), dtype=np.int8)
        self.b = np.zeros((9, 9), dtype=np.int8)
        self.gb = open_grid(walls=[(r, 4) for r in range(0, 8)])

    def test_weights(self):
        u = sample_progress(5)
        np.testing.assert_allclose(u, [0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(source_weights(5, "uniform"), np.ones(5))
        np.testing.assert_allclose(source_weights(5, "linear"), u)
        np.testing.assert_allclose(source_weights(5, "quadratic"), u**2)
        for kind in ("linear", "quadratic"):
            w = source_weights(5, kind)
            self.assertTrue(np.all(np.diff(w) > 0))
        with self.assertRaises(ValueError):
            source_weights(5, "cubic")

    def test_propagation_matches_closed_form(self):
        g = self.gb
        samples = np.array([(0, 0), (0, 3), (3, 3), (6, 3), (8, 3)])
        w = source_weights(len(samples), "linear")
        field, owner = propagate(g, samples, w, 0.02)
        d = np.stack([g.distance_map([g.idx(r, c)]) for r, c in samples])
        closed = np.max(w[:, None, None] - 0.02 * d, axis=0)
        ok = g.free & np.isfinite(field)
        self.assertLess(float(np.max(np.abs(field[ok] - closed[ok]))), 1e-9)
        self.assertEqual(int((owner == -1).sum()), int((~g.free).sum()))

    def test_obstacles_are_nan_and_block_propagation(self):
        g = self.gb
        samples = np.array([(0, 0)])  # single source on the far side of the wall
        field, _ = propagate(g, samples, np.ones(1), 0.02)
        self.assertTrue(np.all(np.isnan(field[~g.free])))
        d = g.distance_map([g.idx(0, 0)])
        # the cell right behind the wall must decay the long way round
        behind = int(d[0, 5])
        self.assertAlmostEqual(float(field[0, 5]), 1.0 - 0.02 * behind, places=9)
        self.assertAlmostEqual(lipschitz_violation(g, field, 0.02), 0.0, places=9)

    def test_repeated_source_cell_keeps_strongest_weight(self):
        g = open_grid()
        samples = np.array([(5, 5), (5, 6), (5, 5)])  # last sample repeats the first cell
        w = np.array([0.0, 0.5, 1.0])
        field, owner = propagate(g, samples, w, 0.02)
        self.assertAlmostEqual(float(field[5, 5]), 1.0)
        self.assertEqual(int(owner[5, 5]), 2)


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Stage1Config()
        cls.scenes = {
            t: generate_scene(i, t, cls.cfg.seed + i, cls.cfg) for i, t in enumerate("ABCD")
        }

    def test_scene_is_connected_and_endpoints_are_valid(self):
        for sc in self.scenes.values():
            g = Grid(sc["occupancy"])
            md = sc["metadata"]
            self.assertEqual(g.components()[1], 1)
            self.assertTrue(g.is_free(*md["start"]))
            self.assertTrue(g.is_free(*md["goal"]))
            self.assertGreaterEqual(md["stats"]["geodesic_sg"], self.cfg.min_geodesic_sg)

    def test_every_grade_is_reachable_on_every_scene_type(self):
        for sc in self.scenes.values():
            g = Grid(sc["occupancy"])
            s, t = tuple(sc["metadata"]["start"]), tuple(sc["metadata"]["goal"])
            prof = detour_profile(g, s, t)
            pool = build_candidate_pool(g, s, t, prof["d_sg"], prof["detour"], self.cfg, np.random.default_rng(0))
            sel = select_by_grade(pool)
            for name, *_ in BUCKETS:
                self.assertIn(name, sel, msg=f"{sc['metadata']['scene_id']} missing {name}")

    def test_selected_ratios_match_their_band(self):
        sc = self.scenes["D"]
        g = Grid(sc["occupancy"])
        s, t = tuple(sc["metadata"]["start"]), tuple(sc["metadata"]["goal"])
        prof = detour_profile(g, s, t)
        pool = build_candidate_pool(g, s, t, prof["d_sg"], prof["detour"], self.cfg, np.random.default_rng(1))
        sel = select_by_grade(pool)
        for name, lo, hi, _center in BUCKETS:
            cand = sel[name]
            self.assertGreaterEqual(cand["ratio"], lo - 1e-9)
            if hi is not None:
                self.assertLess(cand["ratio"], hi)


if __name__ == "__main__":
    unittest.main()
