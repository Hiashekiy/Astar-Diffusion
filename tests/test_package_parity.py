"""Parity: does the copyable package reproduce the workspace implementation exactly?

The workspace (`src/guidance_field`) is the experiment code that produced the published
numbers; `field-guided-search` is the trimmed deployment package. This test pins them
together so the package cannot silently drift:

* M0 sources           : workspace ``build_all_methods(...)['M0']`` vs package ``trajectory_sources(method='drop')``
* field                : workspace ``propagation.propagate_scipy`` vs package ``propagate_scipy``
* goal-distance model  : workspace ``normalize.manhattan_distance_map`` vs package ``normalise_distance('manhattan')``
* search               : workspace ``priority_search(LinearFusionPolicy)`` vs package ``guided_search``

Lives in the workspace on purpose: the package must not depend on the experiment code.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "field-guided-search" / "src"))

from guidance_field.config import FIELD_CLIP, Config  # noqa: E402
from guidance_field.normalize import normalize_distance as ws_normalize_distance  # noqa: E402
from guidance_field.normalize import prepare_scalars  # noqa: E402
from guidance_field.policies import LinearFusionPolicy, priority_search, astar_search  # noqa: E402
from guidance_field.propagation import propagate_scipy as ws_propagate_scipy  # noqa: E402
from guidance_field.sources import build_all_methods  # noqa: E402
from guidance_field.trajectories import generate_trajectories  # noqa: E402
from guidance_field.field import source_weights as ws_source_weights  # noqa: E402
from guidance_field.scenes import generate_scene  # noqa: E402
from guidance_field.grid import Grid as WsGrid  # noqa: E402

import field_guided_search as pkg  # noqa: E402
from field_guided_search.search import normalise_distance as pkg_normalize_distance  # noqa: E402


def build_case(map_type: str, index: int):
    cfg = Config()
    scene = generate_scene(index, map_type, cfg.seed + index, cfg)
    grid_ws = WsGrid(scene["occupancy"])
    meta = scene["metadata"]
    start, goal = tuple(meta["start"]), tuple(meta["goal"])
    rng = np.random.default_rng(cfg.seed + index)
    produced = generate_trajectories(grid_ws, scene["occupancy"], start, goal, cfg, rng)
    return cfg, grid_ws, start, goal, produced["trajectories"]


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grids = {}   # scene key -> (cfg, ws_grid, pkg_grid, start, goal, trajectories)
        for map_type in ("A", "C"):
            cfg, grid_ws, start, goal, trajectories = build_case(map_type, 0)
            cls.grids[map_type] = (cfg, grid_ws, pkg.Grid(grid_ws.occupancy), start, goal, trajectories)

    # ------------------------------------------------------------------ #
    def test_sources_match_m0(self):
        for map_type, (cfg, grid_ws, _, _, _, trajectories) in self.grids.items():
            for grade in ("feasible", "moderate_collision"):
                raw = trajectories[grade]["raw_path"]
                ws = build_all_methods(grid_ws, raw, cfg)["M0"]
                pk = pkg.trajectory_sources(grid_ws, raw, num_samples=cfg.num_source_samples,
                                            method="drop")
                np.testing.assert_array_equal(np.asarray(ws["positions_rc"]), pk["positions_rc"],
                                              err_msg=f"{map_type}/{grade} positions")
                np.testing.assert_allclose(np.asarray(ws["progress"]), pk["progress"],
                                           err_msg=f"{map_type}/{grade} progress")

    def test_field_matches(self):
        for map_type, (cfg, grid_ws, _, _, _, trajectories) in self.grids.items():
            for grade in ("feasible", "severe_collision"):
                raw = trajectories[grade]["raw_path"]
                ws_src = build_all_methods(grid_ws, raw, cfg)["M0"]
                pk_src = pkg.trajectory_sources(grid_ws, raw, num_samples=cfg.num_source_samples,
                                                method="drop")
                for weight in ("linear", "quadratic"):
                    ws_field = ws_propagate_scipy(grid_ws, np.asarray(ws_src["positions_rc"]),
                                                  ws_source_weights(ws_src["progress"], weight), cfg.lam)
                    pk_field = pkg.propagate_scipy(pkg.Grid(grid_ws.occupancy),
                                                   pk_src["positions_rc"],
                                                   pkg.weight_at_progress(pk_src["progress"], weight),
                                                   cfg.lam)
                    ok = grid_ws.free & np.isfinite(ws_field) & np.isfinite(pk_field)
                    self.assertGreater(ok.sum(), 0)
                    err = float(np.max(np.abs(ws_field[ok] - pk_field[ok])))
                    self.assertLess(err, 1e-12, msg=f"{map_type}/{grade}/{weight}: {err:.3e}")
                    self.assertEqual(int((np.isnan(ws_field) != np.isnan(pk_field)).sum()), 0)

    def test_manhattan_distance_matches(self):
        for map_type, (cfg, grid_ws, pkg_grid, start, goal, _) in self.grids.items():
            ws_d, ws_max = ws_normalize_distance(grid_ws, goal)
            pk_d, pk_max = pkg_normalize_distance(pkg_grid, goal, mode="manhattan")
            self.assertAlmostEqual(ws_max, pk_max, places=9, msg=map_type)
            free = grid_ws.free
            err = float(np.max(np.abs(ws_d[free] - pk_d[free])))
            self.assertLess(err, 1e-12, msg=f"{map_type}: d_hat differs by {err:.3e}")

    def test_search_matches_workspace_policies(self):
        """Same sources, same field, same priority formula -> identical search trace."""
        for map_type, (cfg, grid_ws, pkg_grid, start, goal, trajectories) in self.grids.items():
            for grade in ("feasible", "moderate_collision", "severe_collision"):
                raw = trajectories[grade]["raw_path"]
                ws_src = build_all_methods(grid_ws, raw, cfg)["M0"]
                pk_src = pkg.trajectory_sources(grid_ws, raw, num_samples=cfg.num_source_samples,
                                                method="drop")
                weight = "linear"
                ws_field = ws_propagate_scipy(grid_ws, np.asarray(ws_src["positions_rc"]),
                                              ws_source_weights(ws_src["progress"], weight), cfg.lam)
                scalars = prepare_scalars(grid_ws, ws_field, weight, goal)
                pk_field = pkg.propagate_scipy(pkg_grid, pk_src["positions_rc"],
                                               pkg.weight_at_progress(pk_src["progress"], weight),
                                               cfg.lam)
                for alpha in (0.0, 0.25, 0.5, 1.0):
                    ws_res = priority_search(grid_ws, start, goal,
                                             LinearFusionPolicy(scalars, alpha=alpha))
                    pk_res = pkg.guided_search_grid(pkg_grid, start, goal, pk_field, alpha=alpha,
                                               weight=weight, clip=FIELD_CLIP[weight],
                                               distance="manhattan")
                    label = f"{map_type}/{grade}/alpha={alpha}"
                    self.assertEqual(ws_res["status"], pk_res["status"], msg=label)
                    self.assertEqual(ws_res["expanded"], pk_res["expanded"],
                                     msg=f"{label}: expanded {ws_res['expanded']} vs {pk_res['expanded']}")
                    if ws_res["status"] == "success":
                        self.assertEqual(len(ws_res["path"]), inst_len(pk_res["path"]), msg=label)

    def test_astar_matches(self):
        for map_type, (cfg, grid_ws, pkg_grid, start, goal, _) in self.grids.items():
            ws_res = astar_search(grid_ws, start, goal)
            pk_res = pkg.astar(pkg_grid, start, goal)
            self.assertEqual(ws_res["status"], pk_res["status"], msg=map_type)
            self.assertEqual(int(ws_res["g_cost"]), int(pk_res["length"]), msg=map_type)
            self.assertEqual(int(ws_res["expanded"]), int(pk_res["expanded"]), msg=map_type)


def inst_len(path) -> int:
    arr = np.asarray(path)
    return 0 if arr.size == 0 else len(arr)


if __name__ == "__main__":
    unittest.main()
