"""Field construction for stage 2A: legal sources -> weighted multi-source propagation.

Delegates to the stage-1 ``propagate`` unchanged; stage 2's only responsibility is that
every source handed over is a free-space cell and that each weight comes from the
source's *original* progress ``u`` (never from its index in the surviving list).
"""

from __future__ import annotations

import numpy as np

from guidance_stage1.fields import weight_at_progress
from .propagation import propagate as propagate_field
from guidance_stage1.grid import Grid
from guidance_stage2.config import WEIGHT_KINDS


def source_weights(progress: np.ndarray, kind: str) -> np.ndarray:
    """Weights from the stored original progress, without re-normalisation."""
    return np.asarray(weight_at_progress(np.asarray(progress, dtype=np.float64), kind), dtype=np.float64)


def build_weighted_field(grid: Grid, sources: dict, kind: str, lam: float,
                         method: str = "auto") -> dict:
    """One field for one (source set, weighting) pair."""
    positions = np.asarray(sources["positions_rc"], dtype=int).reshape(-1, 2)
    progress = np.asarray(sources["progress"], dtype=np.float64).reshape(-1)
    if len(positions) == 0:
        return {
            "field": np.full((grid.height, grid.width), np.nan, dtype=np.float64),
            "owner": np.full((grid.height, grid.width), -1, dtype=np.int32),
            "weights": np.zeros(0),
            "num_sources": 0,
            "empty": True,
        }
    weights = source_weights(progress, kind)
    field, owner = propagate_field(grid, positions, weights, lam, method=method,
                                       return_owner=True)
    return {
        "field": np.asarray(field, dtype=np.float64),
        "owner": owner,
        "weights": weights,
        "num_sources": int(len(positions)),
        "empty": False,
    }


def build_all_weight_variants(grid: Grid, sources: dict, lam: float) -> dict[str, dict]:
    return {kind: build_weighted_field(grid, sources, kind, lam) for kind in WEIGHT_KINDS}
