# Guidance Field — Stage 1

Mechanism check for one question only:

> Given a feasible trajectory from S to G, can trajectory sampling + weighted
> multi-source propagation turn a 1-D trajectory into a 2-D guidance field that
> carries global spatial structure?

**No learning, no search, no diffusion.** Every trajectory in this stage comes from
traditional graph search (`A*`, weighted `A*`, greedy best-first, waypoint-constrained
`A*`), and the only outputs are fields, figures and statistics. Search speed-up is
stage 2.

## Quick start

```bash
pip install -r requirements.txt

python scripts/run_stage1.py           # 20 maps x 4 quality grades x 3 weightings (~45 s)
python scripts/check_acceptance.py     # audit the artifacts against the 7 criteria
python -m unittest discover -s tests   # 16 unit tests (~2 s)
```

Useful flags: `--quick` (one map per scene type), `--no-figures`, `--no-lambda-sweep`,
`--outdir`, `--lam`, `--samples`.

## The mechanism

For a trajectory resampled at `M = 32` equal-arc points `s_0..s_31` with strength `w_i`
and decay `lambda`:

```
F(x) = max_i [ w_i - lambda * d_free(x, s_i) ]        d_free = obstacle-aware geodesic distance
```

Implemented as one max-Dijkstra seeded with all 32 sources, so walls block propagation
exactly as they block movement. Three weightings are compared on every trajectory:

| Field | weights | intent |
|---|---|---|
| A | `w_i = 1` | plain multi-source distance field (baseline) |
| B | `w_i = u_i` | linear ramp 0→1 from S to G |
| C | `w_i = u_i^2` | reinforces the goal end |

## Scenes

Four non-perfect maze families, `64 x 64`, 5 random instances each (seeds are fixed, so
runs are reproducible):

| Type | name | why it exists |
|---|---|---|
| A | `dual_corridor` | two parallel channels; the lower one is longer by a forced 2·dy step |
| B | `detour_block` | one large central block; top bypass is direct, bottom bypass carries a bump |
| C | `dead_end_branches` | crooked corridor + tempting dead ends + bypass loops |
| D | `multi_branch_maze` | 31x31 lattice with loops, rooms, isolated blocks and pockets |

S and G are placed by the generator that knows the topology (level and between the two
corridor mouths for A/B, corridor ends for C, farthest pair for D). Every instance is
validated: single free component, `d(S,G) >= 45`, free fraction in `[0.03, 0.80]`, and
`1 + max_detour/L* >= 1.35` — i.e. a Poor-grade detour must actually exist.

## Trajectory quality

Quality is graded by the **measured path-cost ratio** `rho = L(P) / L(P*)`, never by the
name of the algorithm that produced it:

| grade | rho |
|---|---|
| excellent | 1.00 – 1.05 |
| good | 1.05 – 1.15 |
| medium | 1.15 – 1.30 |
| poor | > 1.30 |

Each map gets a pool of ~70–90 feasible trajectories (`A*`, weighted `A*` at 1.15/1.5/2/3,
greedy best-first, heuristic-perturbed `A*` for near-optimal-but-different routes, 48
waypoint `A*` candidates biased toward deep pockets, and 24 two-waypoint chains). One
candidate per grade is selected by distance to the grade centre, preferring trajectories
that do **not** double back — a dead-end excursion retraces cells the field already
covers and would show up much less clearly than a genuine alternative corridor.

## Outputs

```
outputs/guidance_field_stage1/
├── scenes/scene_XXX/{occupancy.npy, start_goal.npy, metadata.json}
├── paths/scene_XXX/{excellent,good,medium,poor}.npy          full trajectory (K,2)
│                    {excellent,...}_samples.npy               the 32 arc-length sources
│                    metadata.json                             rho, length, generator, backtracks
├── fields/scene_XXX/{grade}_{uniform,linear,quadratic}.npy   float32, NaN on obstacles
├── figures/scene_XXX/{grade}.png                             the 4-panel figure
│                     trajectories.png                        all 4 grades on one map
│                     comparison.png                          3 weightings x 4 qualities
│                     owners.png                              dominant-source partition
│                     profiles.png                            F(s_i) along the trajectory
├── figures/scene_gallery.png                                 all 20 scenes x 4 grades
├── figures/{summary.png, lambda_sensitivity.png}
├── analysis/lambda_sensitivity.json
├── summary.json                                              config, aggregates, per-scene records
├── acceptance.json / acceptance.txt                          the 7 criteria with evidence
```

## Module map

| file | role |
|---|---|
| `config.py` | every tunable and grade boundary, in one frozen dataclass |
| `grid.py` | occupancy grid, 4-neighbour geodesics, components, cycle rank, leaf peeling, `detour_profile`, geodesic balls |
| `scenes.py` | the four generators, carving helpers, structural validation |
| `paths.py` | `A*`/weighted/greedy/waypoint search, arc-length resampling, candidate pool, grade selection |
| `fields.py` | weight families and the weighted multi-source propagation, plus all field measurements |
| `viz.py` | all figures |
| `pipeline.py` | orchestration and JSON output |
| `acceptance.py` | the 7 criteria, re-derived from the saved files |

`detour_profile` deserves a note: `detour(x) = d(S,x) + d(x,G) - d(S,G)` is 0 exactly on
some optimal route and grows with how far off-route a cell sits. It depends only on the
map, so it is the fair, path-independent label for "correct channel vs. wrong branch".

## Design decisions worth knowing

- **Arc-length, not node index.** Node density varies wildly between an `A*` path and a
  waypoint path; sources must represent equal spatial progress.
- **A trajectory that doubles back legitimately yields repeated source cells.** The
  propagation keeps the strongest weight at a shared cell instead of failing.
- **Geodesic balls for region statistics.** A Euclidean disk around S or G reaches across
  walls and pollutes `F_start`/`F_goal`; the geodesic ball is the meaningful region.
- **Display ranges are per weighting and clipped.** Far from the trajectory every field
  decays to a large negative number; letting that tail set the colour scale makes the
  0→1 ramp invisible. Raw extremes stay in the data, the colorbar marks the clip.
- **`lambda` scale.** The useful regime is `lambda * L ~ 1..4`. Below that the strongest
  source floods the whole map (at `lambda = 0.005` a single source owns 98% of the cells),
  above it the field collapses toward nearest-source distance. `0.02` for `L = 60..190`
  keeps 28–32 of the 32 sources owning part of the map. See `analysis/lambda_sensitivity.json`.

## What stage 1 established

All 7 acceptance criteria pass (see `outputs/guidance_field_stage1/acceptance.txt`):
fields are stable and exactly equal to the closed form, walls block propagation,
strengths are monotone, a continuous high-response band follows every trajectory, field
divergence tracks trajectory divergence (`r = 0.886`), off-route pockets are separated
from the trajectory's own channel (AUC = 1.0), and the goal weighting turns a flat
distance field into a directional one (`delta_F: -0.001 → +0.985`).

Full discussion, numbers and caveats: [`docs/stage1-report.md`](docs/stage1-report.md).

## Next stage (not implemented here)

`Global Guidance Field -> A* / Focal Search`: use `F` as a heuristic/prior, measure node
expansion, direction agreement and end-to-end planning time against the plain geometric
heuristic. The honest caveat from stage 1 is that the field faithfully encodes whatever
trajectory it is given — it will not repair a bad one — so stage 2 needs a robustness
mechanism rather than blind trust in the prior.
