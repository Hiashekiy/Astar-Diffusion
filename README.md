# guidance-field

Trajectory → obstacle-aware 2-D guidance field → guided grid search.

A diffusion/demonstration trajectory is turned into a **global potential field** over the
map, and that field is then used as the priority of a best-first search. The library is
self-contained: it generates its own maps, trajectories and fields, and never needs
precomputed data.

```
trajectory : (T,2) polyline  ->  sources (drop/project/bridge)  ->  F(x)  ->  best-first search
```

---

## Repository layout

| path | what it is |
|---|---|
| repository root (`src/guidance_field`, `tests/`, `docs/`, `scripts/`) | **development workspace** - the full library used for the study. Not a package: nothing to install, just put `src` on `PYTHONPATH`. |
| `field-guided-search/` | the **distributable package** (`field-guided-search`, module `field_guided_search`): map + guidance path -> field -> guided search, 5 modules, 19 tests. This is the folder to copy elsewhere. |

## Running the workspace

There is no install step. From the repository root:

```bash
# Windows PowerShell
$env:PYTHONPATH="src"
python -m unittest discover -s tests          # 56 tests
python -m guidance_field.cli demo --out out/  # CLI still works
```

```bash
# bash / zsh
PYTHONPATH=src python -m unittest discover -s tests
```

Requirements: `numpy`, `scipy`, `matplotlib` (see `requirements.txt`).

## Quick start (Python)

```python
from guidance_field import Config, build_case, build_weighted_field, prepare_scalars
from guidance_field import astar_search, priority_search, FieldOnlyPolicy, LinearFusionPolicy
from guidance_field.sources import build_all_methods

cfg  = Config()                                    # 64x64, lambda=0.02, 64 sources
case = build_case(0, "C", cfg.seed, cfg)           # one map + 4 graded trajectories
grid, start, goal = case["grid"], case["start"], case["goal"]

raw = case["trajectories"]["moderate_collision"]["raw_path"]   # a colliding demo
src = build_all_methods(grid, raw, cfg)["M0"]                  # legal free-space sources
field = build_weighted_field(grid, src, "linear", cfg.lam)["field"]

sc = prepare_scalars(grid, field, "linear", goal)              # F_hat, d_hat
for name, pol in (("A*", None),
                  ("Field-Only", FieldOnlyPolicy(sc)),
                  ("Linear Fusion", LinearFusionPolicy(sc, alpha=0.5))):
    res = astar_search(grid, start, goal) if pol is None else priority_search(grid, start, goal, pol)
    print(name, res["status"], res["expanded"], len(res["path"]))
```

## Quick start (CLI)

```bash
$env:PYTHONPATH="src"                          # PowerShell; bash: export PYTHONPATH=src
python -m guidance_field.cli demo --out out/                 # one map + comparison + 2 PNGs
python -m guidance_field.cli experiment --maps 2 --out out/  # study -> CSV + summary.json
python -m guidance_field.cli field --map map.npy --trajectory traj.npy --out field.png
```

`demo` on a 64x64 maze with a moderately colliding trajectory (measured, not illustrative):

| method | status | expanded | path | ms | vs A\* |
|---|---|---:|---:|---:|---:|
| A\* (baseline) | success | 1105 | 146 | 2.78 | 1.00x |
| Distance-GBFS | success | 416 | 150 | 0.56 | 0.38x |
| Field-Only | success | 178 | 150 | 0.26 | 0.16x |
| Linear Fusion alpha=0.5 | success | 241 | 146 | 0.37 | 0.22x |

---

## The three layers

### 1. Field — `field.py`, `propagation.py`

```
F(x) = max_i [ w(u_i) - lam * d_free(x, s_i) ]
```

* `w(u)` is `uniform` (1), `linear` (u) or `quadratic` (u^2); `u` is each source's
  **original** trajectory progress, never re-normalised.
* `d_free` is the 4-connected geodesic distance; obstacles stay `NaN`, so the field never
  leaks through walls.
* `lam = 0.02`: the field is directional (it decays along the trajectory) but keeps its
  structure across the map; the measured useful range is `lam * L ~ 1..4`.

Four interchangeable propagators, `propagation.propagate(method=...)`:

| method | algorithm | accuracy | when to use |
|---|---|---|---|
| `scipy` (default) | virtual super-source + `scipy.sparse.csgraph.dijkstra` on a cached CSR graph | exact (<= 4e-12) | any map, fastest |
| `vectorized` | synchronous max-plus sweeps over the whole grid | exact (bit-identical) | wide/open maps, reference implementation |
| `heap` | multi-source max-Dijkstra | exact | only when the dominant source is needed |
| `bucket` | Dial queue on `C = min[d + (1-w)/lam]` | <= lam/(2*scale) | comparative baseline |

Measured on the 20 reference maps: `scipy` 6.2 ms total vs `heap` 141.8 ms (**22.7x**) and
vs `vectorized` 49.1 ms (7.9x). `vectorized` is 20x faster than the heap on open maps but up
to 16x *slower* on long-corridor mazes, because it needs one sweep per unit of geodesic
diameter. Numbers and methodology: `docs/results.md`.

### 2. Sources — `sources.py`

A demonstrated trajectory may pass through obstacles, which is not a legal source set.
Three repairs, all keeping the original `u`:

| id | rule |
|---|---|
| `M0` | delete samples inside obstacles, keep the rest |
| `M1` | move each offending sample to the nearest legal cell within radius 3 |
| `M2` | bridge each collision interval with a local A\* inside a ROI, keep inherited progress |

### 3. Search — `policies.py`

| id | priority `J(n)` | uses the field |
|---|---|---|
| `B0_astar` | `g + h` | no (shortest-path baseline) |
| `B1_distance_gbfs` | `d_hat_G` | no |
| `B2_field_only` | `-F_hat` | yes |
| `B3_linear_fusion` | `alpha*d_hat_G - (1-alpha)*F_hat` | yes |
| `B4_nonlinear_fusion` | `alpha*d_hat_G^gamma - (1-alpha)*F_hat`, gamma=0.5 | yes |
| `B5_adaptive_fusion` | `alpha(n)` from 0.25 far to 0.75 near | yes |

`F_hat` uses the shared clip table `config.FIELD_CLIP`; `d_hat_G = d_G / D_max`. B1-B5 build
their priority from **precomputed scalar maps only** — `g(n)` is never passed in, so they
cannot silently degrade into A\*. Only B0 claims shortest-path optimality.

---

## Project layout

```
src/guidance_field/
    config.py        Config dataclass + all shared constants (one source of truth)
    grid.py          occupancy grid, geodesic distance/components, geodesic balls
    paths.py         A*, greedy best-first, resampling, path-quality grading
    collision.py     supercover segment tracing, collision intervals, projection
    scenes.py        four map families (A clutter, B rooms, C maze, D structured)
    trajectories.py  shortcut / Bezier / directed trajectories, graded by measured collision
    sources.py       M0 / M1 / M2 source handling
    propagation.py   heap / vectorized / bucket / scipy backends + auto
    field.py         weights and weighted multi-source field construction
    normalize.py     F_hat and d_hat normalisation for the search layer
    policies.py      B0-B5 priority policies, best-first loop, Dijkstra reference
    metrics.py       source / field / channel metrics, aggregation, baseline comparison
    experiment.py    build_case -> run_case -> run_study (CSV + summary.json + figures)
    viz.py           compact plotting (black obstacles, labelled S/G, white halo paths)
    cli.py           `guidance-field demo|experiment|field`
tests/               56 unit + contract tests (stdlib unittest, no pytest needed)
scripts/             propagation benchmark
docs/results.md      measured evidence behind every number in this README
```

## Conventions

* Everything is stored `(row, col)`; matplotlib gets `(x, y) = (col, row)`.
* Obstacles are drawn dark, free space light, `S` blue, `G` red, collision parts of a
  trajectory red, bridges green.
* Timings always report **field build** and **search** separately: a field rebuilt per query
  can dominate the runtime, and hiding that would be misleading.

## Testing

```bash
python -m unittest discover -s tests      # 56 tests, ~6 s
python scripts/benchmark_propagation.py   # propagator benchmark at 64/128/256
```
