# TODO — STL Animation Studio

Improvement plan for the authoring tool (`stl-animation-studio/index.html`) and the
stljax pipeline (`stl-animation-studio/python/`). Ordered roughly by impact.

## 1. Timeline: visualize every spec and its time domain

- [x] One labeled lane per enabled spec (name + type), band spanning its time domain:
  - `reach` → `[t1, t2]`, `together` → `[t1, t2]`
  - `avoid`, `separation` → full `[0, K−1]`
  - `smooth` → `[1, K−2]`
  - `anchor` → two point-markers at `t = 0` and `t = K−1`
  - `precede` → full `[0, K−1]` (Until scope)
- [x] Color the band by satisfaction status (green ρ ≥ 0 / red ρ < 0), matching the card chips.
- [x] Mark the witness timestep (◆) on each lane; clicking it jumps the playhead
  (same behavior as the card "show" button).
- [x] Click a band to select/scroll to its spec card; hover shows the formula + ρ as a tooltip.
- [x] Timeline grows with spec count; scrub track (with t-axis ticks) stays at the bottom.
- [x] Highlight bands that are *active* at the current playhead.

## 2. Easier STL editing

- [x] **Drag time windows on the timeline**: band edges change `t1`/`t2`, band middle
  slides the whole window (reach/together lanes; others have implicit domains).
- [x] **Drag radii on the canvas**: grab a hotspot/avoid circle's rim (a handle shows on
  the selected circle) to resize `r` directly; center-drag still moves it.
- [x] **Scrubbable numbers in the formula**: `t1`, `t2`, `r`, `δ`, `a_max`, `ε` render as
  drag-to-edit tokens; card inputs stay in sync during the gesture.
- [x] **STL mini-DSL**: double-click a formula to edit it as text; Enter parses (window
  from the first `[a,b]`, threshold from the last number — labels like `h1` contain
  digits, so position-based parsing was deliberately avoided), Esc cancels, invalid
  input flashes red.
- [x] **+ bundling** button (together specs were deletable but not creatable).
- [x] Undo/redo (⌘Z / ⇧⌘Z) for spec + group edits, with gesture coalescing so slider
  drags land as one undo step.

## 3. stljax backend for consistency (replace the JS solver)

- [x] Refactored `pipeline.py` into `stl_core.py` (build_terms / optimize_stream /
  exact_rhos / report), `cli.py`, `server.py`; `pipeline.py` kept as a compat shim.
- [x] FastAPI server (`pixi run serve`):
  - `POST /api/verify` — scene JSON → per-spec exact ρ (with spec ids)
  - `POST /api/optimize?steps=N` — SSE stream of intermediate trajectories + final exact ρ
  - serves the static frontend at `/`, so one command runs everything
- [x] Frontend solver toggle **browser | stljax**: stljax mode debounces spec edits,
  streams the solve live, and reports the server-exact min ρ next to the browser's.
- [x] JS solver kept as a low-latency preview *during drag gestures only* in stljax mode;
  streamed server positions are the authoritative result (and are skipped mid-drag so
  they never fight the preview).
- [x] Automated cross-check (`pixi run crosscheck`): numpy port of the browser's exact
  verifier vs stljax, asserted to ≤0.05 px on the demo scene before/after optimization
  (passes with Δ = 0.0000 on all specs). Verified live too: after a 6000-step server
  solve, browser cert and server-exact min ρ agree (+1.3 px both).
- [x] Pixi tasks: `serve`, `demo`, `optimize`, `verify`, `crosscheck`.
- [~] JAX re-trace caching: *documented decision instead* — each request re-jits (~1–2 s
  at K=60) because spec parameters are baked into closures as constants; avoiding
  retrace needs a params-as-arguments refactor that isn't worth it at this scale
  (noted in `stl_core.optimize_stream`'s docstring).

## 4. Path following (user request, added after 1–3)

- [x] **✏ path**: draw a freehand stroke on the canvas → a `follow` spec
  `G[t1,t2] ‖p_i − q(t)‖ ≤ r`, with q(t) the stroke arc-length-resampled across the
  window (constant speed along the drawn path). Window inferred from the stroke's
  horizontal extent; falls back to the full timeline for vertical/reverse strokes.
- [x] Canvas: corridor tube + dashed centerline, moving q(t) marker while active,
  drag-to-translate; full card (window, radius, groups, weight), scrubbable formula,
  DSL editing, timeline lane with draggable window, pointwise-margin plot, witness.
- [x] Backend: `follow` term in `stl_core.build_terms` (windowed Always on the corridor
  margin signal, `resample_polyline` matching the JS resampler); covered by
  `pixi run crosscheck` (Δ = 0.0000) and verified live (browser ρ = server ρ = +2.65).
- [x] Hardening found during this work: `pixi run serve` now uses `--reload`, and the
  browser flags "⚠ server skipped: …" if the backend ever drops an enabled spec
  (that's exactly how a stale server was caught).

## 5. Other suggestions

### Assignment-aligned (GA2 Part 2.4 feature list)
- [x] **Group definition UI**: ⬚ groups lasso — encircle objects on the canvas, assign
  them to an existing or new group; group-scoped specs and precede refs stay valid.
- [x] **Trajectory count control**: N input in the header regenerates synthetic presets
  at 2–60 objects (imported datasets keep their own trajectories).
- [x] **User study instrumentation**: every authoring action is logged with a session
  timestamp; ⎙ log downloads the JSON.
- [x] **Video export**: ⏺ rec captures the canvas via MediaRecorder → webm (mp4 on Safari).
- [ ] Run the informal user study itself (3–5 participants, task + completion time +
  open-ended question) — human task; the tooling above collects the data.

### Verifiability / research story
- [x] **Robustness-over-time plot**: selecting any spec plots its pointwise margin m(t)
  under the timeline (domain shaded, zero line, red/green by sign, playhead cursor).
- [x] **A/B ghost overlay**: ⧉ pin freezes the current trajectories as a dashed overlay —
  e.g. pin the browser solve, then switch to stljax and compare.

### Robustness / polish
- [x] Autosave scene to localStorage (5 s interval + beforeunload); restores on reload
  with a "pick a preset to start fresh" hint.
- [x] Keyboard shortcuts: space play/pause, ←/→ (+shift) playhead step, ⌫ delete selected
  spec, ⌘Z/⇧⌘Z undo/redo, s solver toggle. Documented in the footer.
- [x] Schema unification: `together` specs always carry explicit `t1`/`t2` (normalized on
  import; the Python side already tolerated both).
- [~] Web Worker for the JS solver: *deliberately deferred* — in stljax mode the JS
  solver is preview-only (drag gestures), and in browser mode the adaptive iteration
  budget already caps solver time at ~14 ms/frame. Moving it to a worker would add
  state-sync complexity (full P/constraints mirroring per edit) for little gain now
  that the heavy solves run server-side. Revisit only if large imported datasets
  (>30 objects) need browser-mode solving at 60 fps.
