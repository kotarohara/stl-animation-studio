"""FastAPI backend exposing the stljax pipeline to the browser tool.

Run from the repo root:
  pixi run serve            # uvicorn on http://localhost:8642

Serves the static frontend at / and two endpoints:
  POST /api/verify?which=optimized   scene JSON -> per-spec exact robustness
  POST /api/optimize?steps=6000      scene JSON -> SSE stream of intermediate
                                     trajectories, final event carries exact ρ

The browser's "solver: stljax" mode POSTs the current scene here (debounced on
every spec edit) and animates the streamed positions live.
"""
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import stl_core

ROOT = Path(__file__).resolve().parents[1]   # frontend dir (contains index.html)

app = FastAPI(title="STL Animation Studio backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/verify")
async def verify(request: Request, which: str = "optimized"):
    scene = await request.json()
    rhos = stl_core.exact_rhos(scene, None if which == "input" else which)
    return {
        "specs": rhos,
        "min_rho": min((r["rho"] for r in rhos), default=float("nan")),
    }


@app.post("/api/optimize")
async def optimize(request: Request, steps: int = 6000):
    scene = await request.json()
    steps = max(1, min(int(steps), 40000))

    def gen():   # sync generator: Starlette iterates it in a threadpool
        for ev in stl_core.optimize_stream(scene, steps=steps):
            ev.pop("scene", None)       # the browser only needs positions + ρ
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# static frontend last, so /api/* wins
app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")
