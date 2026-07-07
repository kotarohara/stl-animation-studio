#!/usr/bin/env python
"""Cross-implementation agreement test.

The browser tool hand-rolls exact STL robustness in JS; the pipeline uses stljax.
This script ports the browser's verifier to numpy (a line-by-line translation of
verifyConstraint in ../index.html) and asserts that it agrees with stljax on the
demo scene — both on the raw input and after an optimization run.

  pixi run crosscheck          # or: pixi run python .../crosscheck.py [steps]

Exit code 0 = every enabled spec's ρ matches within tolerance.
"""
import sys

import numpy as np

import stl_core

TOL = 0.05   # px — stljax runs float32, the JS-style verifier float64


# ---------------------------------------------------------------- JS-port verifier

def js_exact_rhos(scene, which="optimized"):
    """Numpy port of the browser's exact verifier (index.html verifyConstraint)."""
    src = (scene.get(which) if which else None) or [o["points"] for o in scene["objects"]]
    P = np.asarray(src, dtype=float)                      # (N, K, 2)
    K = scene["K"]
    objs = scene["objects"]

    def members(c):
        gs = c.get("groups")
        if not gs:
            return list(range(len(objs)))
        return [i for i, o in enumerate(objs) if o["group"] in gs]

    def group(gid):
        return [i for i, o in enumerate(objs) if o["group"] == gid]

    out = []
    for c in scene["constraints"]:
        if not c.get("enabled", True):
            continue
        t = c["type"]
        if t == "reach":
            d = np.linalg.norm(P[members(c)][:, c["t1"]:c["t2"] + 1] - [c["x"], c["y"]], axis=-1)
            rho = (c["r"] - d).max(axis=1).min()          # min over objects of max over window
        elif t in ("together", "separation"):
            mem = members(c)
            t1 = int(c["t1"]) if c.get("t1") is not None else 0
            t2 = int(c["t2"]) if c.get("t2") is not None else K - 1
            sub = P[mem][:, t1:t2 + 1]                    # (M, T, 2)
            iu = np.triu_indices(len(mem), 1)
            d = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)[iu]
            rho = (c["dmax"] - d).min() if t == "together" else (d - c["dmin"]).min()
        elif t == "smooth":
            acc = P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]
            rho = (c["amax"] - np.linalg.norm(acc, axis=-1)).min()
        elif t == "avoid":
            d = np.linalg.norm(P - [c["x"], c["y"]], axis=-1)
            rho = (d - c["r"]).min()
        elif t == "follow":
            mem = members(c)
            if not mem or len(c.get("pts", [])) < 2:
                continue
            q = stl_core.resample_polyline(c["pts"], c["t2"] - c["t1"] + 1)
            d = np.linalg.norm(P[mem][:, c["t1"]:c["t2"] + 1] - q[None], axis=-1)
            rho = (c["r"] - d).min()
        elif t == "anchor":
            starts = np.array([o["points"][0] for o in objs])
            ends = np.array([o["points"][-1] for o in objs])
            rho = min(
                (c["eps"] - np.linalg.norm(P[:, 0] - starts, axis=-1)).min(),
                (c["eps"] - np.linalg.norm(P[:, -1] - ends, axis=-1)).min(),
            )
        elif t == "precede":
            hs = next((h for h in scene["constraints"]
                       if h.get("id") == c.get("hotspot") and h["type"] == "reach"), None)
            A, B = group(c["first"]), group(c["then"])
            if hs is None or not A or not B:
                continue
            h = np.array([hs["x"], hs["y"]])
            cA = P[A].mean(axis=0)                         # (K, 2)
            cB = P[B].mean(axis=0)
            phi = np.linalg.norm(cB - h, axis=-1) - c["r"]   # B stays out...
            psi = c["r"] - np.linalg.norm(cA - h, axis=-1)   # ...until A arrives
            # ρ(φ U ψ) = max_t min(ψ(t), min_{s≤t} φ(s)) — inclusive overlap
            rho = np.maximum.reduce(np.minimum(psi, np.minimum.accumulate(phi)))
        else:
            continue
        out.append({"id": c.get("id"), "type": t, "rho": float(rho)})
    return out


# ---------------------------------------------------------------- comparison

def compare(scene, which, label):
    a = stl_core.exact_rhos(scene, which)
    b = js_exact_rhos(scene, which)
    assert len(a) == len(b), f"{label}: term count mismatch {len(a)} vs {len(b)}"
    print(f"\n{label}:")
    print(f"  {'spec':<22}{'stljax ρ':>12}{'js-port ρ':>12}{'Δ':>10}")
    worst = 0.0
    for ra, rb in zip(a, b):
        assert ra["id"] == rb["id"], f"{label}: spec order mismatch {ra['id']} vs {rb['id']}"
        delta = abs(ra["rho"] - rb["rho"])
        worst = max(worst, delta)
        flag = "" if delta <= TOL else "   ← MISMATCH"
        print(f"  {ra['name']:<22}{ra['rho']:>+12.3f}{rb['rho']:>+12.3f}{delta:>10.4f}{flag}")
    print(f"  max |Δ| = {worst:.4f} px (tolerance {TOL})")
    return worst


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    scene = stl_core.demo_scene()
    # add a follow (drawn-path corridor) spec threading through both hotspots,
    # so the cross-check also covers the path-following semantics
    scene["constraints"].append(dict(
        id="f1", type="follow", label="f1", r=70, t1=8, t2=52,
        pts=[[70, 290], [285, 265], [545, 300], [750, 300]],
        groups=["A", "B", "C"], weight=1.5, enabled=True))

    worst = compare(scene, None, "raw input trajectories")

    print(f"\noptimizing demo scene ({steps} steps) …")
    scene = stl_core.optimize(scene, steps=steps, every=1000)
    worst = max(worst, compare(scene, "optimized", "optimized trajectories"))

    if worst <= TOL:
        print("\n✓ cross-check passed: stljax and the JS-port verifier agree")
        sys.exit(0)
    print("\n✗ cross-check FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
