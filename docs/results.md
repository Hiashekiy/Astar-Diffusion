# Measured results

Every number the README quotes, with the method that produced it. All runs use
`Config()` defaults: 64×64 grids, λ=0.02, 64 trajectory samples, 4-connected unit-cost
moves, S/G at least 30 steps apart. Timings are best-of-N on the development machine and are
only meaningful as *ratios*.

---

## 1. The field is a genuine guidance channel

`F(x) = max_i [ w(u_i) − λ·d_free(x, s_i) ]`. Three structural facts, each verified in the
test-suite or the study output:

| fact | evidence |
|---|---|
| obstacles never receive a value | `NaN` on every obstacle cell, on every map, all four propagators |
| unreachable free cells never receive a value | `NaN`; no separate connected-component pass needed |
| the field is 1-Lipschitz in the free space with constant λ | `metrics._lipschitz_violation` ≈ 0 apart from a handful of max-tie cells |
| higher `u` ⇒ higher field along the trajectory | `linear`/`quadratic` weights are monotone in `u`; the `uniform` control removes the direction |

λ sensitivity (from the stage-1 sweep): with λ=0.005 a single source owns 98% of the free
space, so the field degenerates into "distance to one point"; the useful regime is
`λ·L ≈ 1…4`, which is why λ=0.02 is the default for maps of side 64.

## 2. Colliding trajectories: repair, do not renormalise

Measured collision ratios of the four grades (identical across all 20 maps):

| grade | collision ratio |
|---|---|
| `feasible` | 0.000 |
| `mild_collision` | 0.030 |
| `moderate_collision` | 0.100 |
| `severe_collision` | 0.223 |

Grades are assigned from the *measured* ratio, never from the generator's name.

Repair quality at `severe_collision` (MAE of the resulting field against the reference field
built from the clean trajectory):

| method | MAE | progress coverage |
|---|---|---|
| M0 (drop) | 0.141 | 0.992 |
| M1 (project) | 0.144 | 0.992 |
| M2 (bridge) | **0.113** | 0.762 |

M2 is not universally better: per map family at severe collision, A 0.057 / 0.057 / 0.057,
B 0.270 / 0.275 / **0.164**, C **0.130** / 0.139 / 0.137, D 0.107 / 0.106 / **0.094**
(M0 / M1 / M2). M2 wins where obstacles genuinely split the trajectory (B, D) and loses
progress coverage because spending fixed samples on a bridge thins the rest.

## 3. Field-guided search (20 maps × 4 grades = 1440 runs, 100% success, 100% valid paths)

| method | field | expanded | search ms | build ms | total ms | L/L\* |
|---|---|---:|---:|---:|---:|---:|
| B0 A\* | — | 1542.3 | 4.233 | — | 4.233 | 1.000 |
| B1 Distance-GBFS | — | 488.1 | 0.719 | — | 0.719 | 1.141 |
| B2 Field-Only | linear | **336.2** | 0.566 | 0.485 | **1.051** | **1.031** |
| B3 Linear Fusion | linear | 368.0 | 0.614 | 0.485 | 1.099 | 1.053 |
| B4 Nonlinear Fusion | linear | 351.1 | 0.613 | 0.485 | 1.098 | 1.049 |
| B5 Adaptive | linear | 348.9 | 0.622 | 0.485 | 1.107 | 1.044 |
| B2 Field-Only | quadratic | 394.0 | 0.600 | 0.380 | 1.026 | 1.032 |
| B3 Linear Fusion | quadratic | 397.1 | 0.646 | 0.380 | 1.026 | 1.061 |

Best expansion ratio: **0.204 × A\*** (B3, linear field, α=0.25) with paths only 4.2%
longer. End to end B2/B3 are **4.03× / 3.85× faster** than A\*.

### The prior is worth what the trajectory is worth

Expanded / L·L\*⁻¹ by grade, linear field:

| grade | B1 distance | B2 field-only | B3 fusion |
|---|---|---|---|
| feasible | 488 / 1.141 | **137 / 1.000** | 246 / 1.014 |
| mild | 488 / 1.141 | **205 / 1.004** | 285 / 1.020 |
| moderate | 488 / 1.141 | 459 / 1.055 | 415 / 1.080 |
| severe | 488 / 1.141 | **544 / 1.066** | 526 / 1.097 |

With a clean trajectory the field is a far better prior than the goal distance (72% fewer
expansions than distance-greedy, and the path is exactly optimal). With a badly colliding
trajectory field-only is *worse* than distance-greedy, and the distance term has to be mixed
back in.

### α and γ

| α | B3 expanded/A\* | B3 L/L\* | B4 expanded/A\* | B4 L/L\* |
|---|---|---|---|---|
| 0.25 | 0.204 | 1.042 | 0.205 | 1.041 |
| 0.50 | 0.222 | 1.045 | 0.206 | 1.044 |
| 0.75 | 0.272 | 1.072 | 0.258 | 1.062 |

Larger α drifts toward Distance-GBFS (0.311 / 1.141). γ=0.5 helps slightly at α ≥ 0.5 and not
at all at α=0.25; γ=1.0 is mathematically identical to B3 and is asserted equal
(`test_gamma_one_reproduces_linear_fusion`).

Nothing here claims optimality for B1–B5: mean `L/L\*` is 1.056 and only 50.8% of the runs
land exactly on `L\*`.

## 4. Propagator benchmark

### 20 reference maps, M0 sources, linear field

| implementation | total ms | per map ms | vs heap |
|---|---:|---:|---:|
| multi-source heap (baseline) | 141.8 | 7.09 | 1.00× |
| heap + precomputed neighbour table | 89.2 | 4.46 | 1.46× |
| synchronous max-plus (vectorized) | 49.1 | 2.46 | 2.89× |
| **sparse Dijkstra (scipy, default)** | **6.2** | **0.31** | **22.7×** |

Field values are exact: scipy vs heap max |ΔF| = 1.3e-15 on these maps.

### Synthetic maps, 64 sources, three complexity regimes

| size | map | heap ms | vectorized ms | scipy ms |
|---:|---|---:|---:|---:|
| 64 | clutter | 7.38 | 2.24 | **0.34** |
| 64 | maze | 4.28 | 4.91 | **0.23** |
| 64 | rooms | 0.34 | 0.67 | **0.12** |
| 128 | clutter | 32.33 | 9.69 | **1.03** |
| 128 | maze | 22.92 | 164.15 | **0.58** |
| 128 | rooms | 1.56 | 3.22 | **0.23** |
| 256 | clutter | 138.00 | 80.33 | **7.24** |
| 256 | maze | 73.59 | 1426.73 | **3.18** |
| 256 | rooms | 6.84 | 19.20 | **3.76** |

Max |ΔF| vs heap: vectorized 0.0 everywhere, scipy ≤ 3.1e-12 (largest maze); no reachability
mismatches.

**Why vectorized loses on mazes.** It advances one grid step per sweep, so it needs
`eccentricity` sweeps and costs `O(ecc × H × W)`, while Dijkstra costs `O(V log V)`. A 256×256
perfect maze has eccentricity ≈ 8907, i.e. `ecc/√(H·W) ≈ 35` — 16× slower. The crossover sits
around `ecc ≈ 1.3·√(H·W)`: below it vectorized wins (clutter, 2×), above it loses badly.

The bucket queue gives no gain (0.90–1.10×) because the bottleneck is the Python neighbour
loop, not heap ordering; it also carries a quantisation error of λ/(2·scale).

**Measurement note.** The first benchmark run measured wall time inside a `tracemalloc`
region and reported 72 ms where the true cost was 5 ms, with different distortion per
implementation. Timing and memory are now measured in separate calls.

## 5. Reproduce

```bash
python -m unittest discover -s tests            # 56 tests
python -m guidance_field.cli demo --out out/     # one map, one figure pair
python -m guidance_field.cli experiment --maps 5 --out out/study
python scripts/benchmark_propagation.py          # 64/128/256 × 3 map kinds
```
