"""Fixed experiment parameters for stage 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# A grid value of 0 means free space, 1 means obstacle.
FREE = 0
OCCUPIED = 1

# Source-weight families. "uniform" is the equal-weight distance field (Field-A),
# "linear" and "quadratic" are the goal-weighted fields (Field-B, Field-C).
WEIGHT_KINDS = ("uniform", "linear", "quadratic")

# Path-quality grades keyed by optimality ratio rho = L(P) / L(P*).
# (name, rho_min, rho_max, selection_center) -- rho_max=None means open ended.
BUCKETS = (
    ("excellent", 1.00, 1.05, 1.02),
    ("good", 1.05, 1.15, 1.10),
    ("medium", 1.15, 1.30, 1.22),
    ("poor", 1.30, None, 1.45),
)

SCENE_TYPES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Stage1Config:
    """Every tunable of the stage-1 run, with the defaults used in the report."""

    outdir: str = "outputs/guidance_field_stage1"
    grid_size: int = 64
    instances_per_type: int = 5
    num_samples: int = 32
    lam: float = 0.02
    seed: int = 20250101
    min_geodesic_sg: int = 45
    waypoint_candidates: int = 48
    ball_radius: int = 3

    def as_dict(self) -> dict:
        return asdict(self)
