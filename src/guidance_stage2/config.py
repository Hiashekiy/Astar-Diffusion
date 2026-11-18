"""Stage-2A parameters. Everything tunable lives here, nothing hard-coded elsewhere."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# G0 is the collision-free reference; the other three grades are assigned by the
# *measured* collision ratio, never by which generator produced the trajectory.
COLLISION_BANDS = {
    "mild_collision": (0.01, 0.05),
    "moderate_collision": (0.05, 0.15),
    "severe_collision": (0.15, 0.30),
}

TRAJECTORY_GRADES = (
    "feasible",
    "mild_collision",
    "moderate_collision",
    "severe_collision",
)

MAP_TYPES = ("A", "B", "C", "D")

MAP_TYPE_NAMES = {
    "A": "random_clutter",
    "B": "rooms_and_corridors",
    "C": "multi_branch_maze",
    "D": "structured_topology",
}

# Free-space fraction band per map family: the four families are deliberately not
# forced into one shared band (a maze is denser than a cluttered open field).
FREE_FRACTION_BANDS = {
    "A": (0.60, 0.88),
    "B": (0.50, 0.80),
    "C": (0.30, 0.62),
    "D": (0.42, 0.78),
}

SOURCE_METHODS = ("M0", "M1", "M2")
WEIGHT_KINDS = ("uniform", "linear", "quadratic")


@dataclass(frozen=True)
class Stage2Config:
    outdir: str = "outputs/guidance_field_stage2"
    grid_size: int = 64
    instances_per_type: int = 5
    seed: int = 20260214

    num_source_samples: int = 64
    lam: float = 0.02
    ball_radius: int = 3

    projection_radius: int = 3
    bridge_roi_margins: tuple[int, ...] = (4, 8, 16, 32)
    collision_detection_step: float = 0.2

    min_geodesic_sg: int = 30
    max_scene_attempts: int = 120

    # trajectory generation effort
    num_shortcut_candidates: int = 90
    num_bezier_candidates: int = 60
    num_directed_candidates: int = 40

    # field / region statistics
    detour_near: float = 2.0  # cells: "near-optimal channel"
    detour_far: float = 10.0  # cells: "off-channel region"

    def collision_bands(self) -> dict:
        return dict(COLLISION_BANDS)

    def as_dict(self) -> dict:
        return asdict(self)
