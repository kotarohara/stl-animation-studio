# STL Animation Studio

A research prototype of **declarative + verifiable trajectory-animation authoring** — the
extension direction to Li et al.'s *RouteFlow* (CHI 2025) where the contribution is the
specification layer, not the solver.

## Run

```bash
python3 -m http.server 8642 --directory stl-animation-studio
# open http://localhost:8642
```

Serve over http (not `file://`) so the bundled datasets can be fetched. The app itself is a
single dependency-free file: `index.html`.

## What it demonstrates

Instead of RouteFlow's procedural pipeline (hierarchical edge bundling → circle packing →
scan-line timing), every animation quality is a **Signal Temporal Logic** formula over
object positions `p_i(t)`:

| Spec | STL formula | RouteFlow analogue |
|---|---|---|
| Hotspot | `F[t1,t2] ‖p_i − h‖ ≤ r  ∀i∈G` | bus stop (converge/diverge) |
| Bundling | `G[t1,t2] ‖p_i − p_j‖ ≤ δmax` | shared bus route |
| Separation | `G[0,K] ‖p_i − p_j‖ ≥ δmin` | seat allocation / anti-occlusion |
| Smoothness | `G[1,K−1] ‖p̈_i‖ ≤ amax` | slow-in/slow-out |
| Position correctness | `‖p_i(0)−s_i‖ ≤ ε ∧ ‖p_i(K−1)−e_i‖ ≤ ε` | endpoint fidelity (ε-tolerance) |
| Avoid region | `G[0,K] ‖p_i − c‖ ≥ r` | *(not expressible in RouteFlow)* |
| Precedence | `(‖c_B−h‖ > r) U[0,K] (‖c_A−h‖ ≤ r)` | *(not expressible in RouteFlow)* |

The precedence spec is a true temporal-logic **Until** over group centroids: group B must
stay clear of the hotspot until group A has arrived. No time windows are given — the
optimizer discovers the ordering from the formula alone.

Two robustness computations run side by side:

- **Optimization** — gradient descent (Adam, annealed, auto-reheated on spec edits) on
  *smooth* robustness: soft-min/soft-max (log-sum-exp) over time, softplus hinges with a
  satisfaction margin, nested softmax weight-routing for Until. This mirrors
  `stljax`/STLCG and is hand-rolled in JS so it runs live (~1,500 iter/s at 15 objects;
  iteration count adapts to a ~14 ms frame budget for larger datasets).
- **Verification** — the *exact* (min/max) robustness ρ of every spec, recomputed live.
  ρ > 0 is a formal certificate that the rendered animation satisfies the formula;
  ρ < 0 comes with a witness (worst object/pair and timestep — "show" jumps the playhead
  there). Until semantics use inclusive overlap, matching stljax.

## Authoring interactions

- **+ hotspot / + avoid region**: click the canvas to place a spec (a sensible time window
  is inferred from the x-position; defaults widen on dense datasets). Drag placed specs
  while the solver runs — the animation re-forms live.
- **+ precedence**: adds an Until ordering spec between two groups at a chosen hotspot.
- Every spec is a card: enable/disable, weight, time window, thresholds, group membership.
- Timeline shows spec windows as bands; scrub to inspect any moment.
- **↑ import / ↓ export**: scene JSON round-trips with the offline stljax pipeline (below).
  Import also accepts a *raw dataset* — a JSON array of `[x,y]` polylines — which is
  arc-length-resampled to K=60 steps, fitted to the canvas, subsampled to ≤30 trajectories,
  and auto-grouped by k-means on start/end positions.
- Presets: *converge→travel→diverge*, *detour around a region*, *staggered windows*,
  *precedence (Until)*, and **bird migration** — RouteFlow's real BirdMap dataset
  (109 GPS trajectories of storks, cranes, and birds of prey; from the RouteFlow
  open-source repo, © its authors, bundled here for research use only).

Synthetic presets pin start/end positions exactly (they are the data); imported real data
instead gets a soft *position-correctness* anchor spec, since real endpoints can violate
separation and the trade-off should be visible and weighted, not hard-coded.

## Offline stljax pipeline (`python/pipeline.py`)

The same scene JSON drives an offline optimizer built on the course stack
(`stljax` + `jax` + `optax`, via the repo's pixi environment):

```bash
pixi run python stl-animation-studio/python/pipeline.py demo -o /tmp/scene.json
pixi run python stl-animation-studio/python/pipeline.py optimize /tmp/scene.json -o /tmp/scene_opt.json
pixi run python stl-animation-studio/python/pipeline.py verify /tmp/scene_opt.json
```

Temporal operators (`Eventually`, `Always`, `Until`) are stljax formulas evaluated on
margin signals; `Until` uses a two-channel signal with column-selecting predicates. The
loss uses stljax's `logsumexp` smooth robustness for Eventually/Until and a per-timestep
hinge surrogate for Always-type specs (better-conditioned gradients; the certificate
always uses exact robustness). The demo scene solves to a full certificate from
straight-line initialization in ~25 s on CPU (30k Adam steps).

**Cross-implementation agreement**: a scene optimized by the pipeline and imported into
the browser (solver off) verifies green, with per-spec ρ matching stljax to rounding —
e.g. `data/demo_stljax_optimized.json`: stljax +4.49/+2.63/+4.68/+4.81/+0.77 vs browser
+4.5/+2.6/+4.7/+4.8/+0.8.

Implementation notes: norms use an ε-stabilized `sqrt(Σv²+1e-9)` (the gradient of
`jnp.linalg.norm` is NaN at zero — straight-line initializations hit this); the smooth
Eventually needs a sharp temperature (over-approximation `ln n / T` must be smaller than
the satisfaction margin, or the hinge pressure switches off while the spec is still
violated).

## Files

- `index.html` — the entire interactive prototype (rendering, live solver, exact verifier, UI)
- `python/pipeline.py` — offline stljax/optax pipeline sharing the scene JSON schema
- `data/birdmap_raw.json` — RouteFlow BirdMap dataset (raw polylines, their preprocessing)
- `data/demo_stljax_optimized.json` — pipeline output used for the cross-verification test
