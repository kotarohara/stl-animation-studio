# STL Animation Studio

A research prototype of **declarative + verifiable trajectory-animation authoring** — the
extension direction to Li et al.'s *RouteFlow* (CHI 2025) where the contribution is the
specification layer, not the solver.

## Run

```bash
pixi run serve          # FastAPI backend + frontend at http://localhost:8642
```

This serves the UI *and* the stljax optimization backend (the header's
**solver: stljax** mode). The frontend alone also works with any static server:

```bash
python3 -m http.server 8642 --directory stl-animation-studio
```

Serve over http (not `file://`) so the bundled datasets can be fetched. The app itself is a
single dependency-free file: `index.html` — only the stljax solver mode needs the backend.

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

- **Optimization** — two interchangeable solvers, selected in the header:
  - *browser*: gradient descent (Adam, annealed, auto-reheated on spec edits) on
    *smooth* robustness: soft-min/soft-max (log-sum-exp) over time, softplus hinges with a
    satisfaction margin, nested softmax weight-routing for Until. This mirrors
    `stljax`/STLCG and is hand-rolled in JS so it runs live (~1,500 iter/s at 15 objects;
    iteration count adapts to a ~14 ms frame budget for larger datasets).
  - *stljax*: the scene is shipped to the FastAPI backend (`pixi run serve`) on every
    spec edit (debounced) and optimized with stljax + optax; intermediate trajectories
    stream back over SSE and animate live. The JS solver then acts only as a
    low-latency preview during drag gestures — the authoritative solve is stljax.
- **Verification** — the *exact* (min/max) robustness ρ of every spec, recomputed live.
  ρ > 0 is a formal certificate that the rendered animation satisfies the formula;
  ρ < 0 comes with a witness (worst object/pair and timestep — "show" jumps the playhead
  there). Until semantics use inclusive overlap, matching stljax. In stljax mode the
  server reports its own exact min ρ after each solve, cross-checking the browser's.

## Authoring interactions

- **+ hotspot / + avoid region**: click the canvas to place a spec (a sensible time window
  is inferred from the x-position; defaults widen on dense datasets). Drag placed specs
  while the solver runs — the animation re-forms live; drag a circle's *rim* to resize it.
- **+ precedence**: adds an Until ordering spec between two groups at a chosen hotspot.
- **+ bundling**: adds an Always-together spec over an editable time window.
- **⬚ groups**: lasso objects on the canvas to reassign them to a group (or a new one) —
  group-scoped specs update immediately.
- Every spec is a card: enable/disable, weight, time window, thresholds, group membership.
  Numbers inside the formula are **drag-scrubbable**, and double-clicking a formula edits
  it as text (the window is read from `[t1,t2]`, the threshold from the last number).
- **Timeline** is a spec Gantt chart: one labeled lane per spec showing its exact time
  domain (implicit ones included — separation/avoid span `[0,K−1]`, smoothness `[1,K−2]`,
  anchors mark only the endpoints), colored by satisfaction. Drag a band's edges to change
  `t1/t2`, its middle to slide the window; ◆ marks the worst-case timestep (click to jump);
  scrub on the bottom track.
- Selecting a spec plots its **pointwise margin m(t)** under the timeline — where and why
  ρ goes negative, not just the worst point.
- **⌘Z / ⇧⌘Z** undo/redo spec and group edits; **⌫** deletes the selected spec;
  **←/→** step the playhead; **N** regenerates synthetic presets at a different scale.
- The scene **autosaves** to localStorage every few seconds and restores on reload.
- **⧉ pin A/B** freezes the current trajectories as a dashed overlay (e.g. compare the
  browser solve against the stljax solve); **⏺ rec** exports the canvas to a video file;
  **⎙ log** downloads a timestamped interaction log (user-study instrumentation).
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

## stljax pipeline (`python/`)

The same scene JSON drives the stljax optimizer, built on the course stack
(`stljax` + `jax` + `optax`, via the repo's pixi environment). The code is split into
`stl_core.py` (formulas, optimizer, exact verifier), `cli.py` (below), and `server.py`
(the FastAPI backend behind `pixi run serve`; `pipeline.py` remains as a shim):

```bash
pixi run demo -o /tmp/scene.json
pixi run optimize /tmp/scene.json -o /tmp/scene_opt.json
pixi run verify /tmp/scene_opt.json
pixi run crosscheck        # stljax vs JS-port exact-ρ agreement test
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
+4.5/+2.6/+4.7/+4.8/+0.8. `pixi run crosscheck` automates this: it ports the browser's
exact verifier to numpy line-by-line and asserts per-spec agreement with stljax (≤0.05 px)
on the demo scene, before and after optimization.

Implementation notes: norms use an ε-stabilized `sqrt(Σv²+1e-9)` (the gradient of
`jnp.linalg.norm` is NaN at zero — straight-line initializations hit this); the smooth
Eventually needs a sharp temperature (over-approximation `ln n / T` must be smaller than
the satisfaction margin, or the hinge pressure switches off while the spec is still
violated).

## Files

- `index.html` — the entire interactive prototype (rendering, live solver, exact verifier, UI)
- `python/stl_core.py` — stljax formulas, streaming optimizer, exact verifier (shared core)
- `python/cli.py` — demo / optimize / verify commands (`python/pipeline.py` is a compat shim)
- `python/server.py` — FastAPI backend: `/api/optimize` (SSE stream), `/api/verify`, static UI
- `python/crosscheck.py` — automated stljax ↔ JS-port exact-robustness agreement test
- `data/birdmap_raw.json` — RouteFlow BirdMap dataset (raw polylines, their preprocessing)
- `data/demo_stljax_optimized.json` — pipeline output used for the cross-verification test
