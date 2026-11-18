# Stage 1 report — trajectory → global guidance field

Experiment: `python scripts/run_stage1.py` (seed 20250101, λ = 0.02, M = 32, 64×64 grids).
Audit: `python scripts/check_acceptance.py`. Artifacts: `outputs/guidance_field_stage1/`.

20 maps (4 scene families × 5 instances) · 80 trajectories · 240 fields · runtime 46 s.

---

## 0. What was asked and what was measured

The stage asks one thing: does `trajectory sampling + weighted multi-source propagation`
produce a field with *global spatial structure*, and does that field respond to the
quality of the input trajectory? Nothing here trains, searches or optimises.

Three weightings are compared on every single trajectory:

```
Field-A  w_i = 1        F(x) = 1 - lambda * min_i d_free(x, s_i)     (plain distance field)
Field-B  w_i = u_i      F(x) = max_i [ u_i - lambda * d_free(x, s_i) ]   (linear ramp)
Field-C  w_i = u_i^2    same, quadratic ramp
```

`u_i = i/(M-1)` is normalised trajectory progress, so `u_0 = 0` at S and `u_31 = 1` at G.

---

## 1. Acceptance results — 7/7

| # | criterion | evidence | verdict |
|---|---|---|---|
| 1 | field stable, no breaks | 240/240 fields finite on every free cell; max Lipschitz excess `1.0e-07`; values inside `[w_max - λ·ecc, w_max]` | PASS |
| 2 | obstacles block propagation | Field-A matches `1 - λ·d_free(x, path)` to `5.7e-08`; Field-B matches `max_i[w_i - λ·d_free]` to `5.9e-08`; corr with geodesic distance `-1.000` vs `-0.930` with straight-line distance | PASS |
| 3 | strength monotone S→G | `w_0 = 0`, `w_31 = 1`, strictly increasing; all 80 sample sets start at S and end at G | PASS |
| 4 | continuous high-response band | worst sag = `0.85` source intervals (tolerance `1.5`); absolutely ≤ `0.12` | PASS |
| 5 | quality → different field | `corr(path divergence, field divergence) = 0.886`; where the two trajectories take different routes the fields differ by ≥ `0.161`; where they share a corridor the difference is ≤ `0.148` | PASS |
| 6 | correct vs wrong channel | AUC(corridor outranks off-route pocket) = `1.000` on all 10 multi-branch maps; median pocket suppression `0.18 – 0.37` field units | PASS |
| 7 | weighting adds directionality | `delta_F` = `-0.001` (A), `+0.985` (B), `+0.996` (C); worst case uniform `abs(delta_F) = 0.0098`, linear min `+0.78` | PASS |

Two criteria were rewritten while building the audit, because the first versions were
**ill posed** and would have produced false confidence:

- *"Pointwise comparison of the excellent and poor field over their exclusive corridors."*
  Invalid: the same 0→1 weight ramp is stretched over trajectories of different length, so
  a region can be late-progress on one trajectory and early-progress on the other. The
  early version reported 2/16 "unfaithful" maps that were in fact correct.
- *"corr(F, -d_own) > corr(F, -d_other)."* Invalid: Field-A **is** `1 - λ·d_own` by
  construction (`r = 1.000` for all 20 maps — tautological), and for Field-B the ramp
  dominates the correlation (`r = 0.08 … 0.89`, occasionally lower than against the other
  path). Replaced by the trajectory-divergence ↔ field-divergence relation, which is the
  claim that actually matters and is falsifiable.

---

## 2. The central comparison (Result C)

Aggregate over 20 maps (mean):

| field | ΔF = F_goal − F_start | along-path slope | distinct owners | owner entropy |
|---|---|---|---|---|
| A uniform | **−0.001** | 0.000 | 32.0 | 0.74 |
| B linear | **+0.985** | +0.999 | 31.5 | 0.73 |
| C quadratic | **+0.996** | +1.022 | 24.0 | 0.63 |

The uniform field is *flat in the direction that matters*: it knows how far you are from
the trajectory, but not which way to run along it. The goal-weighted field keeps the same
spatial reach while adding a full 0 → 1 ramp (slope ≈ 1.0 exactly, i.e. the ramp the
weights prescribe), and every one of the 80 trajectories behaves the same way (uniform
`|ΔF| ≤ 0.0098`, linear `ΔF ≥ 0.78`).

Quadratic weighting does not add more direction — it *concentrates* it: ΔF saturates at
≈ 1.0 but the number of samples that own any part of the map drops from ~31 to ~24
(entropy 0.73 → 0.63). Structure is traded away for goal emphasis. For stage 2 that is a
real trade-off, not a free improvement.

Figures: `figures/scene_XXX/comparison.png` (3 weightings × 4 qualities),
`figures/scene_XXX/owners.png` (which sample owns which region),
`profiles.png` (F(s_i) along the trajectory), `summary.png` (this table as bars).

---

## 3. Does trajectory quality show up in the field? (Result B)

Yes, and quantitatively: field divergence grows with trajectory divergence, `r = 0.886`
(linear fields; `0.965` for the uniform fields). Examples:

| map | path divergence (cells) | field divergence |
|---|---|---|
| scene_016 (D) | 28.7 | 0.472 |
| scene_005 (B) | 31.0 | 0.412 |
| scene_015 (D) | 13.6 | 0.253 |
| scene_010 (C) | 2.1 | 0.070 |
| scene_012 (C) | 1.6 | 0.085 |

The low-divergence cases are the interesting ones and are **not** a defect: in those maps
the poor trajectory stays inside the same corridor as the good one and mostly differs by
doubling back into a dead end. The field therefore differs only slightly. That is
exactly the statement the stage wanted to be able to make:

> The field does not repair a bad trajectory. It faithfully encodes the prior it is given.
> A search that consumes it must therefore carry its own robustness mechanism instead of
> trusting the prior blindly.

The selection step reflects the same finding: among candidates of equal `rho`, trajectories
that retrace their own cells are avoided, because a dead-end excursion re-covers cells the
field already covers and is much less visible than a genuine alternative corridor.

Figures: `figures/scene_XXX/trajectories.png` puts all four grades of one map on a single
panel (the clearest view of "same start/goal, different working conditions"), and
`figures/scene_gallery.png` is the contact sheet of all 20 generated maps with their four
trajectories (rows = scene families A/B/C/D).

---

## 4. Correct channel vs. wrong branch (Result D)

"Wrong branch" is labelled **without reference to any trajectory**: a cell is off-route if
`detour(x) = d(S,x) + d(x,G) − d(S,G) ≥ 10`, i.e. routing through it costs at least 10
extra steps. This makes the comparison fair across the four scene families.

On all 10 multi-branch maps (C, D), for the *excellent* trajectory:

- separation AUC (a corridor cell outranks a pocket cell) = **1.000** in every map;
- median suppression of pocket cells relative to the ramp value that governs them:
  **0.18 … 0.37** in field units (the field's full dynamic range is 1.0).

So a pocket is not merely "a bit darker": the field forms a directional funnel whose
maximum always continues toward the goal side of the junction, which is what a
search guided by `F` would follow.

---

## 5. The one free parameter: λ

`λ` is the only knob; `analysis/lambda_sensitivity.json` sweeps it over 40 fields
(excellent + poor trajectories of all 20 maps):

| λ | ΔF linear | distinct owners | pocket contrast |
|---|---|---|---|
| 0.005 | 0.472 | **1.9** | 0.062 |
| 0.010 | 0.824 | 12.1 | 0.144 |
| **0.020** | **0.980** | **30.2** | **0.358** |
| 0.040 | 0.989 | 31.3 | 0.756 |
| 0.080 | 0.989 | 31.5 | 1.521 |

At `λ = 0.005` a single source owns 98% of the map: the trajectory's spatial structure is
gone and the field degenerates into a one-source potential. The useful regime is
`λ·L ≈ 1…4`, which for `L = 60…190` means `λ ≈ 0.01…0.03`; `0.02` was chosen in the middle
of it. Above `λ ≈ 0.05` the field tends toward nearest-source distance (owner count
saturates) — more contrast, less "global" reach.

---

## 6. Known limitations of this stage

1. **Grid metric only.** 4-neighbour geodesics; no anisotropic or continuous-space field
   yet. Diagonal moves are not modelled (which is consistent for `A*` and propagation, but
   is a modelling choice, not a discovery).
2. **Trajectory quality is graded by length.** `rho` ignores clearance, curvature and
   safety. A trajectory that grazes walls is "excellent" if it is short.
3. **Discrete sampling sag.** Between two sources the field sags below the ideal ramp by up
   to `λ·L/(2M)`; this was measured (0.85 source intervals worst case) and is a property of
   `M = 32`, not an artefact. Raising `M` shrinks it linearly.
4. **Pocket labelling uses a threshold** (`detour ≥ 10`). Shallower wrong branches are
   still suppressed by `λ·depth`, simply not counted as pockets.
5. **No search is run.** Nothing here says the field *helps* `A*`. That is stage 2, and the
   honest expectation from §3 is that it will help only as much as the prior is trustworthy.

---

## 7. What stage 2 inherits

- `F` is a well-defined, obstacle-aware, directionally biased potential on the grid.
- Recommended consumption: `h(n) = max(0, F_max - F(n))`-style shaping, or `F` as a
  focal/priority term — the natural scale is already in field units where `ΔF ≈ 1` spans
  the whole trajectory.
- Robustness must come from the search side (e.g. capped heuristic weight, focal
  re-ordering), because the field will faithfully carry a bad prior.
- `λ` must be adapted to path length (`λ·L ≈ 1…4`) rather than fixed, otherwise long
  trajectories silently lose their structure.
