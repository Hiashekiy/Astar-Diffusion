"""Guidance-field library: trajectory -> 2-D obstacle-aware potential field -> guided search.

Three things, in order:

1. **field**       ``F(x) = max_i [ w(u_i) - lam * d_free(x, s_i) ]`` built from trajectory
                   samples (``field.build_weighted_field``), exact and obstacle-aware.
2. **sources**     turn a *colliding* trajectory into legal free-space sources
                   (``sources.build_all_methods``: M0 drop / M1 project / M2 bridge).
3. **search**      best-first policies on top of the field (``policies``: B0-B5), with
                   ``A*`` kept as the shortest-path baseline.

Quick start::

    from guidance_field import Config, Grid, build_case, build_weighted_field, priority_search

    cfg = Config()
    case = build_case(0, "C", cfg.seed, cfg)          # map + graded trajectories
    raw = case["trajectories"]["moderate_collision"]["raw_path"]
    from guidance_field.sources import build_all_methods
    src = build_all_methods(case["grid"], raw, cfg)["M0"]
    field = build_weighted_field(case["grid"], src, "linear", cfg.lam)["field"]
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import (
    COLLISION_BANDS,
    FIELD_CLIP,
    MAP_TYPES,
    MAP_TYPE_NAMES,
    METHOD_LABELS,
    METHODS,
    SOURCE_METHODS,
    TRAJECTORY_GRADES,
    WEIGHT_KINDS,
    Config,
    field_clip_bounds,
)
from .grid import Grid, detour_profile
from .paths import astar, greedy_best_first, path_length, resample_arc_length
from .collision import detect_collision_intervals, measure_collision_ratio, traversed_obstacle_cells
from .scenes import generate_scene
from .trajectories import generate_trajectories
from .sources import build_all_methods, build_sources_bridge, build_sources_drop, build_sources_project
from .propagation import (
    PROPAGATORS,
    propagate,
    propagate_auto,
    propagate_bucket,
    propagate_heap,
    propagate_scipy,
    propagate_vectorized,
)
from .field import build_all_weight_variants, build_weighted_field, source_weights, weight_at_progress
from .normalize import normalize_distance, normalize_field, prepare_scalars
from .policies import (
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
from .metrics import (
    channel_metrics,
    compare_to_baseline,
    compare_to_reference,
    field_metrics,
    source_metrics,
    summarize,
)
from .experiment import build_case, run_case, run_study

__all__ = [
    "__version__",
    # config
    "Config", "FIELD_CLIP", "field_clip_bounds", "WEIGHT_KINDS", "SOURCE_METHODS",
    "MAP_TYPES", "MAP_TYPE_NAMES", "TRAJECTORY_GRADES", "COLLISION_BANDS",
    "METHODS", "METHOD_LABELS",
    # geometry
    "Grid", "detour_profile", "astar", "greedy_best_first", "path_length", "resample_arc_length",
    "detect_collision_intervals", "measure_collision_ratio", "traversed_obstacle_cells",
    # scenario generation
    "generate_scene", "generate_trajectories",
    "build_all_methods", "build_sources_drop", "build_sources_project", "build_sources_bridge",
    # field
    "PROPAGATORS", "propagate", "propagate_auto", "propagate_scipy", "propagate_vectorized",
    "propagate_heap", "propagate_bucket",
    "build_weighted_field", "build_all_weight_variants", "source_weights", "weight_at_progress",
    "normalize_field", "normalize_distance", "prepare_scalars",
    # search
    "astar_search", "priority_search", "dijkstra_length", "path_is_valid",
    "DistancePolicy", "FieldOnlyPolicy", "LinearFusionPolicy", "NonlinearFusionPolicy",
    "AdaptiveFusionPolicy",
    # measurement + experiment
    "source_metrics", "field_metrics", "channel_metrics", "compare_to_reference",
    "summarize", "compare_to_baseline",
    "build_case", "run_case", "run_study",
]
