"""Stage-3 unit tests: the searcher basics required by the plan (§10 Stage 2B-0) plus
the priority/normalisation properties the experiment depends on.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_stage1.grid import Grid  # noqa: E402
from guidance_stage2.config import FIELD_CLIP  # noqa: E402
from guidance_stage3.normalize import normalize_distance, normalize_field, prepare_scalars  # noqa: E402
from guidance_stage3.search import (  # noqa: E402
    AdaptiveFusionPolicy,
    DistancePolicy,
    FieldOnlyPolicy,
    LinearFusionPolicy,
    NonlinearFusionPolicy,
    astar_search,
    dijkstra_length,
    path_is_valid,
    priority_search,
)


def open_grid(h=12, w=12, walls=()):
    occ = np.zeros((h, w), dtype=np.int8)
    for r, c in walls:
        occ[r, c] = 1
    return Grid(occ)


def scalars_for(grid: Grid, goal, weight="linear", value=None):
    if value is None:
        # a plausible field: high near the goal, decaying with Manhattan distance is not
        # required -- only that values are finite on free cells
        d = normalize_distance(grid, goal)[0]
        field = np.where(grid.free, 1.0 - d, np.nan)
    else:
        field = np.where(grid.free, value, np.nan)
    return prepare_scalars(grid, field, weight, goal)


class SearcherBasics(unittest.TestCase):
    """Plan §10, Stage 2B-0 items 1-6."""

    def test_1_four_neighbour_search_reaches_goal(self):
        g = open_grid()
        r = priority_search(g, (0, 0), (11, 11), DistancePolicy(scalars_for(g, (11, 11))))
        self.assertEqual(r["status"], "success")
        self.assertTrue(path_is_valid(g, r["path"], (0, 0), (11, 11))["valid"])

    def test_2_obstacles_are_not_enterable(self):
        walls = [(r, 5) for r in range(0, 10)]  # wall with a gap at the bottom rows
        g = open_grid(walls=walls)
        r = priority_search(g, (0, 0), (0, 11), DistancePolicy(scalars_for(g, (0, 11))))
        self.assertEqual(r["status"], "success")
        info = path_is_valid(g, r["path"], (0, 0), (0, 11))
        self.assertTrue(info["valid"], msg=info)
        self.assertTrue(all(g.free[divmod(int(i), g.width)] for i in r["path"]))

    def test_3_start_and_goal_are_free(self):
        g = open_grid()
        self.assertTrue(g.free[0, 0] and g.free[11, 11])

    def test_4_all_methods_return_continuous_paths(self):
        g = open_grid(walls=[(r, 6) for r in range(0, 8)])
        goal = (11, 11)
        sc = scalars_for(g, goal)
        policies = [
            DistancePolicy(sc),
            FieldOnlyPolicy(sc),
            LinearFusionPolicy(sc, alpha=0.5),
            NonlinearFusionPolicy(sc, alpha=0.5, gamma=0.5),
            AdaptiveFusionPolicy(sc, alpha_far=0.25, alpha_near=0.75, eta=1.0, gamma=0.5),
        ]
        for pol in policies:
            r = priority_search(g, (0, 0), goal, pol)
            self.assertEqual(r["status"], "success", msg=pol.name)
            self.assertTrue(path_is_valid(g, r["path"], (0, 0), goal)["valid"], msg=pol.name)
        b0 = astar_search(g, (0, 0), goal)
        self.assertTrue(path_is_valid(g, b0["path"], (0, 0), goal)["valid"])

    def test_5_non_g_methods_are_complete_without_truncation(self):
        # an adversarial field: prefer cells far from the goal, so the search must be able
        # to leave high-value regions and still terminate at the goal
        g = open_grid()
        goal = (11, 11)
        d = normalize_distance(g, goal)[0]
        adverse = np.where(g.free, d, np.nan)
        sc = prepare_scalars(g, adverse, "linear", goal)
        for pol in (FieldOnlyPolicy(sc), LinearFusionPolicy(sc, alpha=0.25)):
            r = priority_search(g, (0, 0), goal, pol)
            self.assertEqual(r["status"], "success", msg=f"{pol.name} failed on an adverse field")
            self.assertGreater(r["expanded"], 1)

    def test_6_B0_length_equals_dijkstra(self):
        rng = np.random.default_rng(5)
        for _ in range(4):
            occ = (rng.random((16, 16)) < 0.25).astype(np.int8)
            occ[0, :2] = 0
            occ[15, 14:] = 0
            g = Grid(occ)
            s, t = (0, 0), (15, 15)
            L = dijkstra_length(g, s, t)
            res = astar_search(g, s, t)
            if L is None:
                self.assertNotEqual(res["status"], "success")
            else:
                self.assertEqual(res["status"], "success")
                self.assertAlmostEqual(res["g_cost"], L, places=9)


class PriorityProperties(unittest.TestCase):
    def setUp(self):
        self.g = open_grid()
        self.goal = (11, 11)
        self.sc = scalars_for(self.g, self.goal)

    def test_non_g_priority_depends_only_on_precomputed_maps(self):
        for pol in (DistancePolicy(self.sc), FieldOnlyPolicy(self.sc),
                    LinearFusionPolicy(self.sc, alpha=0.4),
                    NonlinearFusionPolicy(self.sc, alpha=0.4, gamma=0.5),
                    AdaptiveFusionPolicy(self.sc, alpha_far=0.2, alpha_near=0.8, eta=1.0, gamma=0.5)):
            self.assertFalse(any(k in vars(pol) for k in ("g", "g_score", "gscore")), msg=pol.name)
            first = pol.priority(3, 4)
            _ = pol.priority(9, 2)
            self.assertEqual(first, pol.priority(3, 4), msg=f"{pol.name} priority is not a pure function")

    def test_gamma_one_reproduces_linear_fusion(self):
        a = priority_search(self.g, (0, 0), self.goal, LinearFusionPolicy(self.sc, alpha=0.6))
        b = priority_search(self.g, (0, 0), self.goal, NonlinearFusionPolicy(self.sc, alpha=0.6, gamma=1.0))
        self.assertEqual(a["expanded"], b["expanded"])
        self.assertEqual(a["path"], b["path"])

    def test_sqrt_distance_amplifies_near_goal_differences(self):
        lin = LinearFusionPolicy(self.sc, alpha=0.5)
        nl = NonlinearFusionPolicy(self.sc, alpha=0.5, gamma=0.5)
        # with F held equal, the distance term alone decides the ordering
        far_gain_lin = self.sc["d_hat"][0, 0] - self.sc["d_hat"][1, 1]
        far_gain_nl = nl.D_pow[0, 0] - nl.D_pow[1, 1]
        near_gain_lin = self.sc["d_hat"][9, 9] - self.sc["d_hat"][10, 10]
        near_gain_nl = nl.D_pow[9, 9] - nl.D_pow[10, 10]
        self.assertNearlyEqual = None
        self.assertGreater(near_gain_nl / max(near_gain_lin, 1e-12),
                           far_gain_nl / max(far_gain_lin, 1e-12))

    def test_alpha_extremes_approximate_the_pure_methods(self):
        zero = priority_search(self.g, (0, 0), self.goal, LinearFusionPolicy(self.sc, alpha=0.0))
        field = priority_search(self.g, (0, 0), self.goal, FieldOnlyPolicy(self.sc))
        self.assertEqual(zero["expanded"], field["expanded"])
        one = priority_search(self.g, (0, 0), self.goal, LinearFusionPolicy(self.sc, alpha=1.0))
        dist = priority_search(self.g, (0, 0), self.goal, DistancePolicy(self.sc))
        self.assertEqual(one["expanded"], dist["expanded"])


class NormalisationTests(unittest.TestCase):
    def test_field_normalisation_matches_configured_clip(self):
        g = open_grid()
        goal = (11, 11)
        field = np.where(g.free, np.linspace(-3, 3, g.size).reshape(g.height, g.width), np.nan)
        f_hat = normalize_field(field, "linear")
        lo, hi = FIELD_CLIP["linear"]
        self.assertGreaterEqual(float(np.nanmin(f_hat[g.free])), -1e-12)
        self.assertLessEqual(float(np.nanmax(f_hat[g.free])), 1 + 1e-12)
        # clipping: everything at or below the lower bound maps to 0
        self.assertEqual(float(normalize_field(np.array([[lo - 5.0]]), "linear")[0, 0]), 0.0)
        self.assertAlmostEqual(float(normalize_field(np.array([[hi + 5.0]]), "linear")[0, 0]), 1.0)

    def test_distance_normalisation_endpoints(self):
        g = open_grid()
        goal = (5, 7)
        d_hat, d_max = normalize_distance(g, goal)
        self.assertAlmostEqual(float(d_hat[goal]), 0.0, places=12)
        self.assertAlmostEqual(float(d_hat[g.free].max()), 1.0, places=6)
        self.assertGreater(d_max, 0)


class StatusTests(unittest.TestCase):
    def test_unreachable_goal_is_reported(self):
        occ = np.ones((8, 8), dtype=np.int8)
        occ[1:7, 1:3] = 0
        occ[1:7, 5:7] = 0
        g = Grid(occ)
        sc = scalars_for(g, (3, 6), value=0.5)
        r = priority_search(g, (3, 1), (3, 6), LinearFusionPolicy(sc, alpha=0.5))
        self.assertEqual(r["status"], "unreachable")
        self.assertEqual(r["path"], [])

    def test_cutoff_is_distinct_from_unreachable(self):
        g = open_grid()
        goal = (11, 11)
        sc = scalars_for(g, goal)
        r = priority_search(g, (0, 0), goal, LinearFusionPolicy(sc, alpha=0.5), cutoff=3)
        self.assertEqual(r["status"], "cutoff")


if __name__ == "__main__":
    unittest.main()
