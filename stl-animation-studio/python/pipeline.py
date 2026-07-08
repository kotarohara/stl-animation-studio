#!/usr/bin/env python
"""Backwards-compatible entry point — the pipeline now lives in stl_core.py
(core), cli.py (this interface), and server.py (the HTTP backend).

  pixi run python stl-animation-studio/python/pipeline.py demo -o /tmp/scene.json
  pixi run python stl-animation-studio/python/pipeline.py optimize /tmp/scene.json
  pixi run python stl-animation-studio/python/pipeline.py verify /tmp/scene.json
"""
from cli import main

if __name__ == "__main__":
    main()
