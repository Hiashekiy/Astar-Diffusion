"""End-to-end contract: generate a case, build fields, run every search configuration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_field.config import COLLISION_BANDS, METHODS, TRAJECTORY_GRADES, Config
from guidance_field.experiment import CSV_COLUMNS, build_case, run_case
from guidance_field.field import build_weighted_field
from guidance_field.normalize import normalize_distance, normalize_field
from guidance_field.sources import build_all_methods


class CaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()
        cls.case = build_case(0, "C", cls.cfg.seed, cls.cfg)

    def test_four_grades_are_produced_and_ordered(self):
        grades = [g for g in TRAJECTORY_GRADES if g in self.case["trajectories"]]
        self.assertEqual(len(grades), 4)
        ratios = [self.case["trajectories"][g]["metrics"]["collision_ratio"] for g in grades]
        self.assertEqual(ratios, sorted(ratios), msg=f"collision ratios not ordered: {ratios}")
        self.assertEqual(ratios[0], 0.0)

    def test_grades_respect_their_measured_bands(self):
        for grade, (lo, hi) in COLLISION_BANDS.items():
            ratio = self.case["trajectories"][grade]["metrics"]["collision_ratio"]
            self.assertGreaterEqual(ratio, lo - 1e-9, msg=grade)
            self.assertLessEqual(ratio, hi + 1e-9, msg=grade)

    def test_shortest_path_reference_is_consistent(self):
        grid = self.case["grid"]
        self.assertIsNotNone(self.case["L_star"])
        self.assertTrue(grid.free[self.case["start"]])
        self.assertTrue(grid.free[self.case["goal"]])

    def test_sources_are_free_and_ordered(self):
        raw = self.case["trajectories"]["moderate_collision"]["raw_path"]
        sources = build_all_methods(self.case["grid"], raw, self.cfg)
        for name, src in sources.items():
            pos = np.asarray(src["positions_rc"], dtype=int).reshape(-1, 2)
            self.assertTrue(np.all(self.case["grid"].free[pos[:, 0], pos[:, 1]]), msg=name)
            progress = np.asarray(src["progress"], dtype=float)
            self.assertTrue(np.all(np.diff(progress) >= -1e-9), msg=f"{name} progress not monotone")
            self.assertGreaterEqual(progress.min(), -1e-9)
            self.assertLessEqual(progress.max(), 1 + 1e-9)

    def test_dropping_sources_keeps_progress_holes(self):
        """M0 removes obstacle samples instead of re-normalising, so gaps must show up."""
        grid = self.case["grid"]
        raw = self.case["trajectories"]["severe_collision"]["raw_path"]
        sources = build_all_methods(grid, raw, self.cfg)["M0"]
        progress = np.asarray(sources["progress"], dtype=float)
        self.assertLess(len(progress), self.cfg.num_source_samples)
        self.assertGreater(float(np.max(np.diff(progress))), 1.0 / self.cfg.num_source_samples)

    def test_field_is_deterministic_and_masked(self):
        grid = self.case["grid"]
        raw = self.case["trajectories"]["moderate_collision"]["raw_path"]
        src = build_all_methods(grid, raw, self.cfg)["M0"]
        a = build_weighted_field(grid, src, "linear", self.cfg.lam)["field"]
        b = build_weighted_field(grid, src, "linear", self.cfg.lam)["field"]
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(np.isnan(a[~grid.free])))
        self.assertTrue(np.all(np.isfinite(a[grid.free])))

    def test_normalisation_bounds(self):
        grid = self.case["grid"]
        raw = self.case["trajectories"]["moderate_collision"]["raw_path"]
        src = build_all_methods(grid, raw, self.cfg)["M0"]
        field = build_weighted_field(grid, src, "linear", self.cfg.lam)["field"]
        f_hat = normalize_field(field, "linear")
        self.assertGreaterEqual(float(np.nanmin(f_hat[grid.free])), -1e-12)
        self.assertLessEqual(float(np.nanmax(f_hat[grid.free])), 1 + 1e-12)
        d_hat, d_max = normalize_distance(grid, self.case["goal"])
        self.assertAlmostEqual(float(d_hat[self.case["goal"]]), 0.0, places=12)
        self.assertAlmostEqual(float(np.max(d_hat[grid.free])), 1.0, places=6)
        self.assertGreater(d_max, 0)


class RunCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()
        cls.case = build_case(0, "A", cls.cfg.seed, cls.cfg)
        cls.records, cls.traces = run_case(cls.case, cls.cfg, record_traces=True)

    def test_every_method_ran_on_every_grade(self):
        methods = {r["method"] for r in self.records}
        self.assertTrue(set(METHODS).issubset(methods), msg=f"missing: {set(METHODS) - methods}")
        per_weight = 1 + len(self.cfg.alphas) + len(self.cfg.alphas) + 1  # B2 / B3 / B4 / B5
        expected = len(TRAJECTORY_GRADES) * (2 + len(self.cfg.field_weights) * per_weight)
        self.assertEqual(len(self.records), expected)

    def test_records_carry_the_csv_columns(self):
        for col in CSV_COLUMNS:
            self.assertIn(col, self.records[0])

    def test_all_paths_valid_and_statuses_consistent(self):
        for rec in self.records:
            self.assertEqual(rec["status"], "success")
            self.assertTrue(rec["path_valid"], msg=f"{rec['method']} on {rec['grade']}")
            self.assertIsNotNone(rec["path_length"])

    def test_field_methods_expand_less_than_astar(self):
        for grade in TRAJECTORY_GRADES:
            base = next(r for r in self.records if r["grade"] == grade and r["method"] == "B0_astar")
            best = min((r for r in self.records if r["grade"] == grade and r["field"] != "none"),
                       key=lambda r: r["expanded"])
            self.assertLess(best["expanded"], base["expanded"], msg=grade)

    def test_b0_is_optimal_and_others_are_not_claimed_optimal(self):
        for grade in TRAJECTORY_GRADES:
            base = next(r for r in self.records if r["grade"] == grade and r["method"] == "B0_astar")
            self.assertAlmostEqual(base["optimality_ratio"], 1.0, places=9)
        ratios = [r["optimality_ratio"] for r in self.records if r["method"] != "B0_astar"]
        self.assertTrue(all(v >= 1.0 - 1e-9 for v in ratios))

    def test_gamma_one_reproduces_linear_fusion(self):
        from guidance_field.policies import LinearFusionPolicy, NonlinearFusionPolicy, priority_search

        grid = self.case["grid"]
        raw = self.case["trajectories"]["feasible"]["raw_path"]
        src = build_all_methods(grid, raw, self.cfg)["M0"]
        from guidance_field.field import source_weights
        from guidance_field.normalize import prepare_scalars
        from guidance_field.propagation import propagate

        field = propagate(grid, src["positions_rc"], source_weights(src["progress"], "linear"), self.cfg.lam)
        sc = prepare_scalars(grid, field, "linear", self.case["goal"])
        a = priority_search(grid, self.case["start"], self.case["goal"], LinearFusionPolicy(sc, alpha=0.6))
        b = priority_search(grid, self.case["start"], self.case["goal"],
                            NonlinearFusionPolicy(sc, alpha=0.6, gamma=1.0))
        self.assertEqual(a["expanded"], b["expanded"])
        self.assertEqual(a["path"], b["path"])


if __name__ == "__main__":
    unittest.main()
