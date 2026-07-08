#!/usr/bin/env python
"""CLI for the stljax pipeline (see stl_core.py for the actual machinery).

Usage (from the repo root, inside the course pixi env):
  pixi run python stl-animation-studio/python/cli.py demo -o /tmp/scene.json
  pixi run python stl-animation-studio/python/cli.py optimize /tmp/scene.json -o /tmp/scene_opt.json
  pixi run python stl-animation-studio/python/cli.py verify /tmp/scene_opt.json

Or via the pixi tasks: `pixi run demo`, `pixi run optimize <scene>`, `pixi run verify <scene>`.
"""
import argparse
import json

import stl_core


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="generate a synthetic demo scene")
    d.add_argument("-o", "--out", default="demo_scene.json")
    o = sub.add_parser("optimize", help="optimize a scene with stljax + optax")
    o.add_argument("scene")
    o.add_argument("-o", "--out", default=None)
    o.add_argument("--steps", type=int, default=20000)
    v = sub.add_parser("verify", help="report exact robustness of a scene")
    v.add_argument("scene")
    v.add_argument("--input", action="store_true",
                   help="verify the raw input trajectories instead of the optimized ones")
    args = ap.parse_args()

    if args.cmd == "demo":
        scene = stl_core.demo_scene()
        with open(args.out, "w") as f:
            json.dump(scene, f)
        print(f"wrote {args.out} ({len(scene['objects'])} objects, "
              f"{len(scene['constraints'])} specs)")
    elif args.cmd == "optimize":
        with open(args.scene) as f:
            scene = json.load(f)
        stl_core.report(scene, "optimized")
        print("\noptimizing with stljax smooth robustness + optax Adam …")
        scene = stl_core.optimize(scene, steps=args.steps)
        stl_core.report(scene, "optimized")
        out = args.out or args.scene.replace(".json", "_opt.json")
        with open(out, "w") as f:
            json.dump(scene, f)
        print(f"\nwrote {out}  (import this in the browser tool)")
    elif args.cmd == "verify":
        with open(args.scene) as f:
            scene = json.load(f)
        stl_core.report(scene, None if args.input else "optimized")


if __name__ == "__main__":
    main()
