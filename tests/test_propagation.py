"""Unit tests for the three propagators (equivalence, ownership, edge cases)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidance_field.propagation import propagate_heap as propagate_reference  # noqa: E402
from guidance_field.grid import Grid  # noqa: E402
from guidance_field.propagation import (  # noqa: E402
    propagate,
    propagate_auto,
    propagate_bucket,
    propagate_heap,
    propagate_scipy,
    propagate_vectorized,
)

LAM = 0.02


def grid_with(walls, size=12):
    occ = np.zeros((size, size), dtype=np.int8)
    for r, c in walls:
        occ[r, c] = 1
    return Grid(occ)


def sources_on_path(grid: Grid, n=16):
    rng = np.random.default_rng(0)
    free = np.argwhere(grid.free)
    a = free[int(rng.integers(len(free)))]
    dist = grid.geodesic_distance([grid.idx(*a)])
    b = grid.rc(int(np.argmax(np.where(dist > 0, dist, -1))))
    from guidance_field.paths import astar

    nodes = astar(grid, grid.idx(*a), grid.idx(*b))
    pts = np.array([grid.rc(int(i)) for i in nodes], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    sel = np.linspace(0, cum[-1], n)
    samples = np.rint(np.stack([[np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])] for t in sel])).astype(int)
    return samples, np.linspace(0.0, 1.0, n)


class PropagationEquivalence(unittest.TestCase):
    def setUp(self):
        self.grids = [
            grid_with([]),
            grid_with([(r, 5) for r in range(0, 9)]),
            grid_with([(r, c) for r in range(4, 7) for c in range(4, 9)]),
        ]

    def test_heap_matches_stage1_reference(self):
        for g in self.grids:
            samples, w = sources_on_path(g)
            ref, ref_owner = propagate_reference(g, samples, w, LAM, return_owner=True)
            out, owner = propagate_heap(g, samples, w, LAM, return_owner=True)
            ok = g.free & np.isfinite(ref) & np.isfinite(out)
            self.assertLess(float(np.max(np.abs(ref[ok] - out[ok]))), 1e-12)
            np.testing.assert_array_equal(ref_owner[g.free], owner[g.free])

    def test_vectorized_is_bit_exact(self):
        for g in self.grids:
            samples, w = sources_on_path(g)
            ref = propagate_reference(g, samples, w, LAM)
            out = propagate_vectorized(g, samples, w, LAM)
            ok = g.free & np.isfinite(ref) & np.isfinite(out)
            self.assertLess(float(np.max(np.abs(ref[ok] - out[ok]))), 1e-12)
            self.assertEqual(int((np.isnan(ref) != np.isnan(out)).sum()), 0)

    def test_scipy_matches_heap_exactly(self):
        for g in self.grids:
            samples, w = sources_on_path(g)
            ref = propagate_heap(g, samples, w, LAM)
            out = propagate_scipy(g, samples, w, LAM)
            ok = g.free & np.isfinite(ref) & np.isfinite(out)
            self.assertLess(float(np.max(np.abs(ref[ok] - out[ok]))), 1e-10)
            self.assertEqual(int((np.isnan(ref) != np.isnan(out)).sum()), 0)

    def test_auto_uses_exact_scipy_backend(self):
        g = self.grids[1]
        samples, w = sources_on_path(g)
        ref = propagate_heap(g, samples, w, LAM)
        out = propagate_auto(g, samples, w, LAM)
        np.testing.assert_allclose(out, ref, rtol=0.0, atol=1e-10, equal_nan=True)

    def test_invalid_sources_and_lam_are_rejected(self):
        g = self.grids[2]  # this grid has obstacles (rows 4..6 x cols 4..8)
        with self.assertRaises(ValueError):
            propagate_scipy(g, np.array([[5, 5]]), np.array([1.0]), LAM)  # source inside an obstacle
        with self.assertRaises(ValueError):
            propagate_scipy(g, np.array([[0, 0]]), np.array([1.0]), 0.0)  # lam must be > 0
        with self.assertRaises(ValueError):
            propagate_scipy(g, np.array([[0, 0], [1, 1]]), np.array([1.0]), LAM)  # length mismatch

    def test_bucket_error_bounded_by_quantisation(self):
        for scale in (4, 16):
            for g in self.grids:
                samples, w = sources_on_path(g)
                ref = propagate_reference(g, samples, w, LAM)
                out = propagate_bucket(g, samples, w, LAM, scale=scale)
                ok = g.free & np.isfinite(ref) & np.isfinite(out)
                self.assertLess(float(np.max(np.abs(ref[ok] - out[ok]))), LAM / (2 * scale) + 1e-12)

    def test_obstacles_stay_nan(self):
        for fn in (propagate_heap, propagate_vectorized, propagate_bucket):
            g = self.grids[2]
            samples, w = sources_on_path(g)
            out = fn(g, samples, w, LAM)
            self.assertTrue(np.all(np.isnan(out[~g.free])), msg=fn.__name__)

    def test_unreachable_free_cells_are_nan(self):
        occ = np.ones((10, 10), dtype=np.int8)
        occ[1:9, 1:4] = 0
        occ[1:9, 6:9] = 0  # two separate rooms
        g = Grid(occ)
        samples = np.array([[4, 2]])
        w = np.array([1.0])
        for fn in (propagate_heap, propagate_vectorized, propagate_bucket):
            out = fn(g, samples, w, LAM)
            self.assertTrue(np.all(np.isnan(out[1:9, 6:9])), msg=fn.__name__)
            self.assertTrue(np.all(np.isfinite(out[1:9, 1:4])), msg=fn.__name__)

    def test_duplicate_source_cells_keep_max_weight(self):
        g = grid_with([])
        samples = np.array([[5, 5], [5, 6], [5, 5]])
        w = np.array([0.0, 0.5, 1.0])
        for fn in (propagate_heap, propagate_vectorized, propagate_bucket):
            out = fn(g, samples, w, LAM)
            self.assertAlmostEqual(float(out[5, 5]), 1.0, places=9, msg=fn.__name__)

    def test_auto_picks_a_correct_implementation(self):
        for g in self.grids:
            samples, w = sources_on_path(g)
            ref = propagate_reference(g, samples, w, LAM)
            out = propagate_auto(g, samples, w, LAM)
            ok = g.free & np.isfinite(ref) & np.isfinite(out)
            self.assertLess(float(np.max(np.abs(ref[ok] - out[ok]))), 1e-12)

    def test_vectorized_refuses_owner(self):
        g = grid_with([])
        samples, w = sources_on_path(g)
        with self.assertRaises(ValueError):
            propagate_vectorized(g, samples, w, LAM, return_owner=True)

    def test_dispatch_rejects_unknown_method(self):
        g = grid_with([])
        samples, w = sources_on_path(g)
        with self.assertRaises(KeyError):
            propagate(g, samples, w, LAM, method="quantum")


if __name__ == "__main__":
    unittest.main()
