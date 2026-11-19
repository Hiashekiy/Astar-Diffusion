"""Search-stage parameters. Nothing here duplicates a constant owned by another stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Priority-method ids and their human labels (plan §6.1).
METHODS = (
    "B0_astar",
    "B1_distance_gbfs",
    "B2_field_only",
    "B3_linear_fusion",
    "B4_nonlinear_fusion",
    "B5_adaptive_fusion",
)

METHOD_LABELS = {
    "B0_astar": "A*  (g + h)",
    "B1_distance_gbfs": "Distance-GBFS",
    "B2_field_only": "Field-Only",
    "B3_linear_fusion": "Linear Fusion",
    "B4_nonlinear_fusion": "Nonlinear Fusion",
    "B5_adaptive_fusion": "Adaptive Fusion",
}

# Methods whose priority must not touch the accumulated cost g(n) (plan §7.1).
NON_G_METHODS = METHODS[1:]

SEARCH_STATUSES = ("success", "unreachable", "cutoff")


@dataclass(frozen=True)
class SearchConfig:
    """Everything tunable for the first round of the search experiment."""

    stage2_outdir: str = "outputs/guidance_field_stage2"
    outdir: str = "outputs/guidance_field_stage3"

    field_weights: tuple[str, ...] = ("linear", "quadratic")
    source_method: str = "M0"

    alphas: tuple[float, ...] = (0.25, 0.50, 0.75)
    gamma_main: float = 0.5

    alpha_far: float = 0.25
    alpha_near: float = 0.75
    adaptive_eta: float = 1.0
    adaptive_gamma: float = 0.5

    # 0 disables the safety cutoff, which is the default for the first round (plan §7.4)
    node_cutoff: int = 0

    # scene used for the single-scene visual verification (Stage 2B-1)
    core_grade: str = "moderate_collision"
    figure_grades: tuple[str, ...] = (
        "feasible",
        "mild_collision",
        "moderate_collision",
        "severe_collision",
    )

    def as_dict(self) -> dict:
        return asdict(self)
