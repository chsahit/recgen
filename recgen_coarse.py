"""Coarse-ONLY RecGen server — the /coarse_mesh/ surface on its own port (:8050).

Why this exists (obm_lite select_tax_split_0715): the policy's selection tax is
~66% coarse stand-in calls, and the plan is to PREFETCH those calls off the
policy's critical path (fire them during a mint, whose ~4.9s blocks the policy
thread anyway). That only works if a prefetch cannot delay a real mint — and on
the single :8040 server it would, for a reason that is not obvious:

    recgen_fastapi's endpoints are `async def` but call BLOCKING work
    (`run_recgen_*`) directly in the coroutine body. That blocks uvicorn's event
    loop, so the server serializes EVERY request. A prefetched /coarse_mesh/
    would sit in front of a /complete_mesh/ mint and inflate action_secs — i.e.
    prefetch would buy selection time by spending action time, which is exactly
    the trade it is supposed to avoid.

Splitting the coarse surface into a second PROCESS decouples the two queues:
prefetch traffic can never head-of-line-block a mint. It does NOT decouple the
GPU — both processes share one device and time-slice on it — so mints may still
slow somewhat. That residual is a measurement (watch action_secs), not an
assumption.

Cost: a second copy of the pipeline (~20GB). Deliberate, and fine on the H200 —
this is the hazard recgen_fastapi's uvicorn comment warns about, incurred on
purpose rather than by accident.

Everything is imported from recgen_fastapi rather than copied: same pipeline
build, same decode/scene-cache/pickle helpers, same endpoint bodies. Only the
ROUTING differs — /complete_mesh/ is deliberately not served here. Keeping one
implementation means the two servers cannot drift.

Run:  python recgen_coarse.py            # :8050
      python recgen_coarse.py --port 8051
"""

import os
import sys

# Must run BEFORE `import recgen_fastapi`, which carries this same guard and
# would otherwise os.execv() us from halfway through this module's import. The
# loader caches LD_LIBRARY_PATH at process start, so set-and-execv is required;
# doing it here means recgen_fastapi's copy is a no-op by the time it runs.
_env_lib = os.path.join(sys.prefix, "lib")
if _env_lib not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    os.environ["LD_LIBRARY_PATH"] = (
        _env_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    ).rstrip(":")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile

# Builds the pipeline + torch.compile at module scope (~20GB, one copy for THIS
# process). Also defines recgen_fastapi.app with /complete_mesh/ on it; that app
# object is simply never served here.
import recgen_fastapi as rg


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Same synthetic coarse request recgen_fastapi warms with: compiling the
    # sparse-structure denoiser is ~30s cold and must not land on a real
    # request. NB the bench additionally wants ~20-30 real calls before it
    # trusts a latency number (coarse_latency_tax_0714 §8) -- this only covers
    # the compile, not that drift.
    rg._warmup()
    yield


app = FastAPI(
    title="RecGen Coarse API",
    description="Sparse-structure-only mesh reconstruction (/coarse_mesh/), "
                "isolated from /complete_mesh/ so prefetch cannot delay a mint",
    lifespan=_lifespan,
)


@app.middleware("http")
async def _log_request_time(request: Request, call_next):
    """Per-request wall-clock as the caller sees it (minus the response write).

    Tagged [recgen_coarse] so a shared log can be split by server: the whole
    point of this process is to tell coarse traffic and mint traffic apart.
    """
    if request.url.path == "/health":
        return await call_next(request)
    start = time.time()
    response = await call_next(request)
    print(
        f"[recgen_coarse] {request.method} {request.url.path} "
        f"-> {response.status_code} in {time.time() - start:.2f}s"
    )
    return response


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "recgen_coarse_server",
            "scene_cache": dict(rg.SCENE_CACHE_STATS)}


@app.post("/reset_scene_cache/")
async def reset_scene_cache():
    """Bench hook — see recgen_fastapi.reset_scene_cache. Call at run START.

    This process has its OWN scene cache (separate module state is not shared
    across processes), so a bench that resets :8040 does NOT reset this one.
    pol3_replay must reset whichever port actually served the coarse calls, or
    the cross-run contamination ac0e3b1 fixed comes back through this door.
    """
    return await rg.reset_scene_cache()


# The signatures are restated because FastAPI builds its request parsing from
# them; the BODIES delegate, so there is exactly one implementation.
@app.post("/coarse_mesh/")
async def coarse_mesh_endpoint(
    rgb: UploadFile = File(..., description="Full UNMASKED scene RGB image (PNG)"),
    depth: UploadFile = File(..., description="Full UNMASKED scene depth (.png uint16 mm or .npy float32 m)"),
    mask: UploadFile = File(..., description="Per-object binary mask PNG (nonzero = object pixels)"),
    intrinsics: UploadFile = File(..., description="Camera intrinsics .npy (3x3 K or [fx,fy,cx,cy])"),
    grid_resolution: int = 32,
    steps: int = 20,
):
    """See recgen_fastapi.coarse_mesh_endpoint — identical behaviour."""
    return await rg.coarse_mesh_endpoint(
        rgb=rgb, depth=depth, mask=mask, intrinsics=intrinsics,
        grid_resolution=grid_resolution, steps=steps,
    )


@app.post("/coarse_mesh_batch/")
async def coarse_mesh_batch_endpoint(
    rgb: UploadFile = File(..., description="Full UNMASKED scene RGB image (PNG), sent ONCE"),
    depth: UploadFile = File(..., description="Full UNMASKED scene depth (.png uint16 mm or .npy float32 m), sent ONCE"),
    masks: list[UploadFile] = File(..., description="N per-object binary mask PNGs for the SAME frame"),
    intrinsics: UploadFile = File(..., description="Camera intrinsics .npy (3x3 K or [fx,fy,cx,cy])"),
    grid_resolution: int = 32,
    steps: int = 20,
):
    """See recgen_fastapi.coarse_mesh_batch_endpoint — identical behaviour."""
    return await rg.coarse_mesh_batch_endpoint(
        rgb=rgb, depth=depth, masks=masks, intrinsics=intrinsics,
        grid_resolution=grid_resolution, steps=steps,
    )


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Coarse-only RecGen server")
    parser.add_argument("--port", type=int, default=8050,
                        help="Port to serve on (default 8050; :8040 is the full server)")
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"[recgen_coarse] serving /coarse_mesh/ + /coarse_mesh_batch/ on "
          f":{args.port} (no /complete_mesh/ — mints stay on :8040)")
    # Pass the app OBJECT, not "recgen_coarse:app": the import-string form makes
    # uvicorn import this module a second time (it is __main__ here, so the
    # import misses sys.modules), which would build a THIRD copy of the
    # pipeline. Same trap recgen_fastapi documents. reload/workers stay off.
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
