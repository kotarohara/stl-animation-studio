"""Core stljax machinery shared by the CLI (cli.py) and the HTTP backend (server.py).

Scene JSON schema is shared with the browser tool (../index.html): groups, objects
(raw input polylines), optional `optimized` trajectories, and STL constraints.
Temporal operators are stljax formulas evaluated on margin signals; the loss uses
stljax's logsumexp smooth robustness for Eventually/Until and a per-timestep hinge
surrogate for Always-type specs. Certificates always use exact robustness.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stljax.formula import Always, Eventually, Predicate, Until

MARGIN = 2.0          # satisfaction margin (px) targeted by the hinge losses
SMOOTH_MARGIN = 0.5   # margin for the acceleration bound (px / step^2)
HINGE_SCALE = 2.0     # softplus temperature of the hinge
# temperature 8: logsumexp over-approximates max by ln(n)/T ≈ 0.3 px — nearly exact,
# so the hinge pressure does not switch off while a spec is still truly violated
SOFT_KW = dict(approx_method="logsumexp", temperature=8.0)

# Atomic formulas. Specs are evaluated on *margin signals*: a 1-D (or 2-channel
# for Until) signal whose value at t is the spec's pointwise robustness, so
# `m > 0` has robustness exactly m and the temporal operators do the rest.
M_POS = Predicate("m", lambda s: s) > 0.0
PHI_CH0 = Predicate("s", lambda s: s[..., 0]) > 0.0
PSI_CH1 = Predicate("s", lambda s: s[..., 1]) > 0.0


# ---------------------------------------------------------------- margins

def snorm(v):
    """Norm with a NaN-safe gradient at zero (jnp.linalg.norm grad is NaN at 0)."""
    return jnp.sqrt(jnp.sum(v * v, axis=-1) + 1e-9)


def members(scene, c):
    objs = scene["objects"]
    gs = c.get("groups")
    if not gs:
        return list(range(len(objs)))
    return [i for i, o in enumerate(objs) if o["group"] in gs]


def group_members(scene, gid):
    return [i for i, o in enumerate(scene["objects"]) if o["group"] == gid]


def pairs_of(idx):
    return [(idx[a], idx[b]) for a in range(len(idx)) for b in range(a + 1, len(idx))]


def build_terms(scene):
    """Compile each constraint into (exact_fn, soft_fn, weight, margin, name, id).

    Both fns map pos (K, N, 2) -> vector of robustness values (one per
    object/pair; the spec's robustness is their min)."""
    K = scene["K"]
    terms = []

    def temporal(formula, sig_fn, pointwise_interval=None):
        def exact(pos):
            sigs = sig_fn(pos)
            return jax.vmap(lambda s: formula.robustness(s))(sigs)

        if pointwise_interval is not None:
            # Always-type specs: hinge every timestep in the loss instead of the
            # temporal soft-min — same surrogate the browser solver uses; the
            # min over timesteps is recovered by the exact verifier above.
            a, b = pointwise_interval

            def soft(pos):
                return sig_fn(pos)[:, a:b + 1].reshape(-1)
        else:
            def soft(pos):
                sigs = sig_fn(pos)
                return jax.vmap(lambda s: formula.robustness(s, **SOFT_KW))(sigs)

        return exact, soft

    for c in scene["constraints"]:
        if not c.get("enabled", True):
            continue
        t = c["type"]
        name = f"{t}:{c.get('label', c.get('id', ''))}"
        cid = c.get("id")

        if t == "reach":
            mem = jnp.array(members(scene, c))
            h = jnp.array([c["x"], c["y"]])
            r = c["r"]

            def sig_fn(pos, mem=mem, h=h, r=r):
                d = snorm(pos[:, mem, :] - h)                      # (K, M)
                return (r - d).T                                   # (M, K)

            f = Eventually(M_POS, interval=[int(c["t1"]), int(c["t2"])])
            ex, so = temporal(f, sig_fn)
            terms.append((ex, so, c["weight"], MARGIN, name, cid))

        elif t in ("together", "separation"):
            prs = jnp.array(pairs_of(members(scene, c)))
            thr = c["dmax"] if t == "together" else c["dmin"]
            sign = 1.0 if t == "together" else -1.0
            interval = (
                [int(c["t1"]), int(c["t2"])] if c.get("t1") is not None else None
            )

            def sig_fn(pos, prs=prs, thr=thr, sign=sign):
                d = snorm(
                    pos[:, prs[:, 0], :] - pos[:, prs[:, 1], :]
                )                                                  # (K, P)
                return (sign * (thr - d)).T                        # (P, K)

            f = Always(M_POS, interval=interval)
            ex, so = temporal(f, sig_fn, pointwise_interval=interval or (0, K - 1))
            terms.append((ex, so, c["weight"], MARGIN, name, cid))

        elif t == "smooth":
            amax = c["amax"]

            def sig_fn(pos, amax=amax):
                acc = pos[2:] - 2 * pos[1:-1] + pos[:-2]           # (K-2, N, 2)
                return (amax - snorm(acc)).T    # (N, K-2)

            f = Always(M_POS)
            ex, so = temporal(f, sig_fn, pointwise_interval=(0, K - 3))
            terms.append((ex, so, c["weight"], SMOOTH_MARGIN, name, cid))

        elif t == "avoid":
            h = jnp.array([c["x"], c["y"]])
            r = c["r"]

            def sig_fn(pos, h=h, r=r):
                d = snorm(pos - h)                                 # (K, N)
                return (d - r).T                                   # (N, K)

            f = Always(M_POS)
            ex, so = temporal(f, sig_fn, pointwise_interval=(0, K - 1))
            terms.append((ex, so, c["weight"], MARGIN, name, cid))

        elif t == "anchor":
            eps = c["eps"]
            starts = jnp.array([o["points"][0] for o in scene["objects"]])
            ends = jnp.array([o["points"][-1] for o in scene["objects"]])

            def fn(pos, eps=eps, starts=starts, ends=ends):
                d0 = snorm(pos[0] - starts)
                d1 = snorm(pos[-1] - ends)
                return eps - jnp.concatenate([d0, d1])             # (2N,)

            terms.append((fn, fn, c["weight"], MARGIN, name, cid))

        elif t == "precede":
            hs = next(
                (h for h in scene["constraints"]
                 if h.get("id") == c.get("hotspot") and h["type"] == "reach"),
                None,
            )
            A = jnp.array(group_members(scene, c["first"]))
            B = jnp.array(group_members(scene, c["then"]))
            if hs is None or A.size == 0 or B.size == 0:
                continue
            h = jnp.array([hs["x"], hs["y"]])
            r = c["r"]

            def sig_fn(pos, A=A, B=B, h=h, r=r):
                cA = pos[:, A, :].mean(axis=1)                     # (K, 2)
                cB = pos[:, B, :].mean(axis=1)
                phi = snorm(cB - h) - r         # B stays out...
                psi = r - snorm(cA - h)                            # ...until A arrives
                return jnp.stack([phi, psi], axis=-1)[None]        # (1, K, 2)

            f = Until(PHI_CH0, PSI_CH1)
            ex, so = temporal(f, sig_fn)
            terms.append((ex, so, c["weight"], MARGIN, name, cid))

    return terms


# ---------------------------------------------------------------- verify

def positions_of(scene, which="optimized"):
    src = (scene.get(which) if which else None) or [o["points"] for o in scene["objects"]]
    return jnp.array(src, dtype=jnp.float32).transpose(1, 0, 2)     # (K, N, 2)


def exact_rhos(scene, which="optimized"):
    """Exact (min/max) robustness of every enabled spec: [{id, name, weight, rho}]."""
    pos = positions_of(scene, which)
    return [
        {"id": cid, "name": name, "weight": w, "rho": float(jnp.min(exact(pos)))}
        for exact, _, w, _, name, cid in build_terms(scene)
    ]


def report(scene, which="optimized"):
    rhos = exact_rhos(scene, which)
    print(f"\nExact STL robustness ({which or 'objects'}):")
    worst = np.inf
    for r in rhos:
        worst = min(worst, r["rho"])
        mark = "✓" if r["rho"] >= 0 else "✗"
        print(f"  {mark}  ρ = {r['rho']:+8.2f}   {r['name']} (w={r['weight']})")
    print(
        f"  → {'CERTIFICATE: all specs satisfied' if worst >= 0 else 'VIOLATED'}"
        f" (min ρ = {worst:+.2f} px)"
    )
    return worst


# ---------------------------------------------------------------- optimize

def optimize_stream(scene, steps=6000, lr=0.5, every=250):
    """Optimize the scene, yielding progress events (a generator so the HTTP
    backend can stream intermediate trajectories to the browser over SSE).

    Yields {"type": "progress", step, total, loss, positions} every `every` steps
    and finally {"type": "done", total, specs, min_rho, positions, scene}.

    Note: each call re-jits the loss (~1-2 s at K=60) because spec parameters are
    baked into the closures as constants. At this scale that is cheaper than the
    params-as-arguments refactor that avoiding retrace would require.
    """
    K, N = scene["K"], len(scene["objects"])
    init = scene.get("optimized") or [o["points"] for o in scene["objects"]]
    pos = jnp.array(init, dtype=jnp.float32).transpose(1, 0, 2)    # (K, N, 2)
    data = jnp.array(
        [o["points"] for o in scene["objects"]], dtype=jnp.float32
    ).transpose(1, 0, 2)

    terms = build_terms(scene)
    pinned = bool(scene.get("pinned", True))
    mask = jnp.ones((K, N, 1))
    if pinned:
        mask = mask.at[0].set(0.0).at[-1].set(0.0)

    def loss_fn(p):
        total = 0.0
        for _, soft, w, margin, _, _ in terms:
            rho = soft(p)
            total += w * jnp.sum(
                HINGE_SCALE * jax.nn.softplus((margin - rho) / HINGE_SCALE)
            )
        return total

    sched = optax.exponential_decay(lr, transition_steps=2000, decay_rate=0.7, end_value=0.06)
    opt = optax.adam(sched)
    state = opt.init(pos)

    @jax.jit
    def step(p, s):
        l, g = jax.value_and_grad(loss_fn)(p)
        updates, s = opt.update(g * mask, s, p)
        return optax.apply_updates(p, updates), s, l

    def as_points(p):
        return np.round(np.asarray(p).transpose(1, 0, 2), 2).tolist()

    if pinned:  # keep the pinned endpoints exactly on the data
        pos = pos.at[0].set(data[0]).at[-1].set(data[-1])
    for k in range(steps):
        pos, state, l = step(pos, state)
        if k % every == 0 or k == steps - 1:
            yield {"type": "progress", "step": k + 1, "total": steps,
                   "loss": float(l), "positions": as_points(pos)}

    out = dict(scene)
    out["optimized"] = as_points(pos)
    rhos = exact_rhos(out, "optimized")
    yield {"type": "done", "total": steps, "specs": rhos,
           "min_rho": min((r["rho"] for r in rhos), default=float("nan")),
           "positions": out["optimized"], "scene": out}


def optimize(scene, steps=1500, lr=0.5, every=300, log=print):
    """Blocking wrapper around optimize_stream for the CLI."""
    final = None
    for ev in optimize_stream(scene, steps=steps, lr=lr, every=every):
        if ev["type"] == "progress" and log:
            log(f"  step {ev['step']:5d}  soft loss {ev['loss']:10.2f}")
        final = ev
    return final["scene"]


# ---------------------------------------------------------------- demo scene

def demo_scene(seed=1):
    """Synthetic converge→travel→diverge scene matching the browser preset."""
    rng = np.random.default_rng(seed)
    K, W, H = 60, 820, 560
    groups = [
        dict(id="A", color="#4fc3f7", count=5, start=dict(x=70, y=115), end=dict(x=750, y=95)),
        dict(id="B", color="#ffb74d", count=5, start=dict(x=70, y=290), end=dict(x=750, y=295)),
        dict(id="C", color="#b388ff", count=5, start=dict(x=70, y=465), end=dict(x=750, y=495)),
    ]

    def sample(cx, cy, r, placed, mind=16):
        for _ in range(400):
            a, rr = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform())
            p = [cx + rr * np.cos(a), cy + rr * np.sin(a)]
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= mind for q in placed):
                placed.append(p)
                return p
        placed.append([cx, cy])
        return [cx, cy]

    objects, starts, ends = [], [], []
    for g in groups:
        for _ in range(g["count"]):
            s = sample(g["start"]["x"], g["start"]["y"], 38, starts)
            e = sample(g["end"]["x"], g["end"]["y"], 36, ends)
            amp = rng.uniform(-18, 18)
            ts = np.linspace(0, 1, K)
            pts = np.stack(
                [s[0] + ts * (e[0] - s[0]),
                 s[1] + ts * (e[1] - s[1]) + amp * np.sin(np.pi * ts)],
                axis=-1,
            )
            objects.append(dict(group=g["id"], points=np.round(pts, 2).tolist()))

    constraints = [
        dict(id="h1", type="reach", label="h1", x=285, y=265, r=30, t1=15, t2=23,
             groups=["A", "B", "C"], weight=3, enabled=True),
        dict(id="h2", type="reach", label="h2", x=545, y=300, r=30, t1=36, t2=44,
             groups=["A", "B", "C"], weight=3, enabled=True),
        dict(id="bun", type="together", dmax=90, t1=19, t2=40,
             groups=["A", "B", "C"], weight=1.2, enabled=True),
        dict(id="sep", type="separation", dmin=12, weight=2.5, enabled=True),
        dict(id="smo", type="smooth", amax=4.5, weight=2.0, enabled=True),
    ]
    return dict(version=1, K=K, canvas=[W, H], pinned=True,
                groups=groups, objects=objects, constraints=constraints)
