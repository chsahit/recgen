"""FastAPI server for RecGen single-view mesh reconstruction.

Mirrors the /complete_mesh/ surface of MINSC's sam3d_fastapi.py: takes a full
scene RGB-D plus a per-object mask and intrinsics, returns a pickled mesh
dict with `vertices`, `faces`, `vertex_colors`.

Frame note: sam3d_fastapi returns the mesh in PyTorch3D camera frame
(-X, -Y, +Z relative to standard pinhole). RecGen returns standard pinhole
camera frame (i.e. result.mesh, the same frame as posed_mesh.obj). Callers
that previously consumed sam3d output may need a frame flip.
"""

import os
import sys

# Ensure the pixi env's libstdc++ wins over /lib64/libstdc++.so.6 (which is
# missing GLIBCXX_3.4.29 and breaks open3d's libLerc.so.4). The dynamic loader
# caches LD_LIBRARY_PATH at process start, so set-and-execv is required —
# mutating os.environ alone won't affect already-running dlopen resolution.
_env_lib = os.path.join(sys.prefix, "lib")
if _env_lib not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    os.environ["LD_LIBRARY_PATH"] = (
        _env_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    ).rstrip(":")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import hashlib
import io
import pickle
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

# Give inductor a cache dir that survives restarts, so the startup warmup below is a
# cache hit (~10s) instead of a cold compile (~30s). Only a fallback: if the
# environment already sets TORCHINDUCTOR_CACHE_DIR this is a no-op. It matters where
# it doesn't (e.g. in the container), since inductor otherwise defaults to
# /tmp/torchinductor_$USER, which doesn't outlive the container. Must precede
# `import torch`.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torchinductor_cache"),
)

import fast_simplification
import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from sklearn.neighbors import KDTree

from recgen_inference import build_recgen, generate, generate_coarse


# ── Model loading (module-level, done once) ───────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_NAME = os.environ.get("RECGEN_PIPELINE", "recgen_base.multiview_stereo")
pipeline = build_recgen.build(PIPELINE_NAME)

# The sparse-structure denoiser is ~60% of a /coarse_mesh/ request: the sampler runs
# it ~35x per request (one forward per step, twice per step inside the CFG interval).
# It is compute-bound at batch 1, so batching the CFG pair buys nothing, but inductor's
# fusion is worth ~1.6x. Shapes are fixed across requests (cond is always 1x1369x1024,
# latent 1x8x16^3), so dynamic=False compiles exactly one graph.
# Set RECGEN_COMPILE=0 to fall back to eager.
#
# NOTE: don't be tempted to switch on TF32 here — allow_tf32/matmul_precision("high")
# measured *50% slower* on this model (it pushes the fp32 head off the fp16 kernels).
COMPILE = os.environ.get("RECGEN_COMPILE", "1") != "0"
if COMPILE and DEVICE == "cuda":
    pipeline.models["sparse_structure_pose_flow_model"] = torch.compile(
        pipeline.models["sparse_structure_pose_flow_model"], dynamic=False
    )


# ── Input loaders (match sam3d_fastapi semantics) ─────────────────────────────

def _decode_depth(data: bytes, filename: str) -> np.ndarray:
    """Decode a depth map as float32 meters with 0 where invalid.

    Accepts .npy (float32 meters) or .png (uint16 millimeters following the
    existing capture-zip convention), dispatched on `filename`'s extension.
    Non-finite pixels are zeroed out.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".npy":
        depth = np.load(io.BytesIO(data)).astype(np.float32)
    else:
        arr = np.array(Image.open(io.BytesIO(data)))
        if arr.dtype == np.uint16:
            depth = arr.astype(np.float32) / 1000.0
        else:
            depth = arr.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def _decode_intrinsics(data: bytes) -> np.ndarray:
    """Decode intrinsics as a 3x3 K matrix. Accepts 3x3 K or 4-vector [fx,fy,cx,cy]."""
    K = np.load(io.BytesIO(data))
    if K.shape == (3, 3):
        return K.astype(np.float64)
    if K.shape == (4,):
        fx, fy, cx, cy = (float(v) for v in K)
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
    raise ValueError(f"intrinsics must be 3x3 or length-4, got {K.shape}")


# ── Decimation (mirrors sam3d_fastapi.py) ─────────────────────────────────────

# TARGET_FACES = 10000
TARGET_FACES = 1500

# Backup caps: b3d_fastapi pads every mesh up to these bounds for JIT shape
# stability and will reject anything above. Step-9 face-count decimation targets
# TARGET_FACES but doesn't bound vertex count, so a second pass here guarantees
# both caps are met before the bundle ships.
# NOTE: bumped 2x for sam3_server.py pipeline (preserves more mesh detail).
# If you route b3d_registration_notebook.py through this endpoint, bump the
# matching pad constants in b3d_fastapi.py or those requests will be rejected.
# MAX_VERTS_PER_MESH = 5200
# MAX_FACES_PER_MESH = 11000
# PAD_DECIMATE_TARGET_FACES = 6500
MAX_VERTS_PER_MESH = 800
MAX_FACES_PER_MESH = 1800
PAD_DECIMATE_TARGET_FACES = 1100


def _vertex_cluster_decimate(verts, faces, max_v, max_f):
    """Always-succeeds fallback: voxel-bin verts and merge until under caps.
    Used when fast_simplification hits a structural floor (non-manifold edges,
    boundary loops, etc.) and refuses to reduce further.
    """
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    extent = o3d_mesh.get_axis_aligned_bounding_box().get_extent()
    diag = float(np.linalg.norm(extent))
    voxel = max(diag / 100.0, 1e-6)
    for _ in range(25):
        out = o3d_mesh.simplify_vertex_clustering(
            voxel_size=voxel,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
        v_out = np.asarray(out.vertices)
        f_out = np.asarray(out.triangles)
        if f_out.shape[0] == 0:
            raise RuntimeError(f"vertex clustering collapsed to 0 faces (voxel={voxel:.4f})")
        if v_out.shape[0] <= max_v and f_out.shape[0] <= max_f:
            return v_out.astype(np.float32), f_out.astype(np.int32)
        voxel *= 1.5
    raise RuntimeError(
        f"vertex clustering failed to converge: last {v_out.shape[0]}v/{f_out.shape[0]}f "
        f"(voxel={voxel:.4f}, caps {max_v}v/{max_f}f)"
    )


def _enforce_mesh_caps(mesh_dict):
    """Second-pass decimation: if a mesh exceeds MAX_VERTS/MAX_FACES, decimate
    until both caps are satisfied. fast_simplification can silently undershoot
    target_count (and hits a hard floor on tricky topology), so retry with
    rising agg, then fall back to open3d vertex clustering as a last resort.
    """
    verts = np.asarray(mesh_dict["vertices"], dtype=np.float32)
    faces = np.asarray(mesh_dict["faces"], dtype=np.int32)
    colors = np.asarray(mesh_dict["vertex_colors"], dtype=np.float32)
    n_v, n_f = verts.shape[0], faces.shape[0]
    if n_v <= MAX_VERTS_PER_MESH and n_f <= MAX_FACES_PER_MESH:
        return mesh_dict

    orig_v, orig_f = n_v, n_f
    target = PAD_DECIMATE_TARGET_FACES
    verts_out, faces_out = verts, faces
    n_v_out, n_f_out = n_v, n_f
    method = None
    for attempt in range(6):
        agg = min(7 + attempt, 10)
        verts_out, faces_out = fast_simplification.simplify(
            verts.astype(np.float64),
            faces.astype(np.int64),
            target_count=target,
            agg=agg,
        )
        n_v_out, n_f_out = verts_out.shape[0], faces_out.shape[0]
        if n_v_out <= MAX_VERTS_PER_MESH and n_f_out <= MAX_FACES_PER_MESH:
            method = f"fast_simplification(target={target},agg={agg},attempts={attempt+1})"
            break
        target = max(int(target * 0.7), 50)

    if method is None:
        # fast_simplification refused to reduce further — hit a structural floor.
        # Vertex clustering merges by spatial bin, ignores topology, always succeeds.
        verts_out, faces_out = _vertex_cluster_decimate(
            verts, faces, MAX_VERTS_PER_MESH, MAX_FACES_PER_MESH,
        )
        method = "vertex_clustering"

    tree = KDTree(verts)
    _, nn_idx = tree.query(np.array(verts_out), k=1)
    new_colors = colors[nn_idx.flatten()].astype(np.float32)
    print(f"  _enforce_mesh_caps: {orig_v}v/{orig_f}f -> {verts_out.shape[0]}v/{faces_out.shape[0]}f via {method}")
    return {
        "vertices": np.asarray(verts_out, dtype=np.float32),
        "faces": np.asarray(faces_out, dtype=np.int32),
        "vertex_colors": new_colors,
    }


def _decimate_mesh_dict(mesh_dict: dict) -> dict:
    """Two-pass decimation matching sam3d_fastapi.run_sam3d_single:
    first reduce to TARGET_FACES if exceeded, then enforce MAX_VERTS/MAX_FACES.
    """
    n_faces = len(mesh_dict["faces"])
    if n_faces > TARGET_FACES:
        orig_vertices = mesh_dict["vertices"]
        orig_colors = mesh_dict["vertex_colors"]
        verts_out, faces_out = fast_simplification.simplify(
            orig_vertices.astype(np.float64),
            mesh_dict["faces"].astype(np.int64),
            target_count=TARGET_FACES,
        )
        tree = KDTree(orig_vertices)
        _, nn_idx = tree.query(np.array(verts_out), k=1)
        new_colors = orig_colors[nn_idx.flatten()]
        print(f"  decimated {n_faces} -> {len(faces_out)} faces")
        mesh_dict = {
            "vertices": np.array(verts_out, dtype=np.float32),
            "faces": np.array(faces_out, dtype=np.int32),
            "vertex_colors": np.array(new_colors, dtype=np.float32),
        }
    return _enforce_mesh_caps(mesh_dict)


# ── Mesh extraction ───────────────────────────────────────────────────────────

def _trimesh_to_dict(mesh: trimesh.Trimesh) -> dict:
    """Convert a trimesh.Trimesh to the {vertices, faces, vertex_colors} dict.

    vertex_colors is (N, 3) float32 in [0, 1] (RGB). Falls back to mid-gray if
    the mesh has no per-vertex colors (e.g. TextureVisuals only).
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    colors_uint8 = getattr(mesh.visual, "vertex_colors", None)
    if colors_uint8 is None or len(colors_uint8) != len(verts):
        colors = np.full((len(verts), 3), 0.5, dtype=np.float32)
    else:
        colors = np.asarray(colors_uint8, dtype=np.float32)[:, :3] / 255.0

    return {"vertices": verts, "faces": faces, "vertex_colors": colors}


class _DegenerateCompletion(Exception):
    """Raised when a completion decodes to no usable geometry (endpoint returns 422)."""


def _drop_nonfinite(mesh_dict: dict, n_voxels: int = -1) -> dict:
    """Drop non-finite vertices and any face touching them.

    The SLAT mesh decoder emits NaN vertices when it runs on a near-empty latent:
    a low-evidence object can leave only a hundred-odd voxels alive through the
    sparse-structure stage, and FlexiCubes' zero-crossing interpolation divides by
    the SDF difference across an edge (`_linear_interp`, no epsilon), which goes to
    NaN on degenerate cells. Nothing downstream tolerates that — the first thing to
    notice is the KDTree in `_decimate_mesh_dict`, which raises a bare
    "Input contains NaN" with no hint of where it came from.

    Repairing here is only right when the NaNs are a minority of an otherwise real
    mesh. If they aren't, the completion is junk and the caller needs to know that,
    so an empty survivor set raises rather than returning a plausible-looking husk.
    """
    verts = np.asarray(mesh_dict["vertices"], dtype=np.float32)
    faces = np.asarray(mesh_dict["faces"], dtype=np.int32)
    colors = np.asarray(mesh_dict["vertex_colors"], dtype=np.float32)

    ctx = f"n_voxels={n_voxels}" if n_voxels >= 0 else "n_voxels=unknown"
    if len(verts) == 0 or len(faces) == 0:
        raise _DegenerateCompletion(
            f"decoder returned an empty mesh ({len(verts)} verts / {len(faces)} faces, {ctx})"
        )

    finite_v = np.isfinite(verts).all(axis=1)
    if finite_v.all():
        return mesh_dict

    faces = faces[finite_v[faces].all(axis=1)]
    if len(faces) == 0:
        raise _DegenerateCompletion(
            f"every face touches a non-finite vertex "
            f"({int((~finite_v).sum())}/{len(verts)} verts are NaN/inf, {ctx})"
        )

    # Reindex onto just the vertices a surviving face still references.
    used = np.zeros(len(verts), dtype=bool)
    used[faces.reshape(-1)] = True
    remap = (np.cumsum(used) - 1).astype(np.int32)
    print(
        f"  _drop_nonfinite: dropped {int((~finite_v).sum())} non-finite verts "
        f"({len(verts)}v/{len(mesh_dict['faces'])}f -> {int(used.sum())}v/{len(faces)}f, {ctx})"
    )
    return {
        "vertices": verts[used],
        "faces": remap[faces].astype(np.int32),
        "vertex_colors": colors[used],
    }


def run_recgen_single(image, depth_map, mask, K) -> dict:
    """Run RecGen on one object given the full scene RGB-D plus a mask.

    Args:
        image:     (H, W, 3) uint8 — full scene RGB (PIL or numpy).
        depth_map: (H, W) float32 meters — full scene depth (0 where invalid).
        mask:      (H, W) bool/uint8 — non-zero where this object's pixels live.
        K:         (3, 3) camera intrinsics.

    Returns a mesh dict in standard-pinhole camera frame.
    """
    start = time.time()

    rgb = np.asarray(image)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    elif rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)

    if not mask_u8.any():
        raise RuntimeError("mask has no non-zero pixels")
    if not np.any(np.isfinite(depth_map) & (depth_map > 0) & mask_u8.astype(bool)):
        raise RuntimeError("no valid depth pixels under the mask")

    result = generate(
        pipeline,
        image=rgb,
        depth=depth_map.astype(np.float32),
        mask=mask_u8,
        intrinsics=np.asarray(K, dtype=np.float64),
        formats=["mesh"],  # skip gaussian / radiance-field decoders — only mesh is used
    )

    n_voxels = len(result._outputs.get("coords", ()))
    mesh_dict = _trimesh_to_dict(result.mesh)
    mesh_dict = _drop_nonfinite(mesh_dict, n_voxels)
    mesh_dict = _decimate_mesh_dict(mesh_dict)
    elapsed = time.time() - start
    print(
        f"run_recgen_single done in {elapsed:.2f}s "
        f"(verts={len(mesh_dict['vertices'])}, faces={len(mesh_dict['faces'])})"
    )
    return mesh_dict


def run_recgen_coarse(image, depth_map, mask, K, grid_resolution: int = 32, steps: int = 20) -> dict:
    """Run only RecGen's coarse (sparse-structure) stage on one object.

    Same inputs as :func:`run_recgen_single`, but skips the SLAT sampling +
    mesh decoder. The occupancy voxels are turned into an untextured mesh via
    marching cubes (downsampled to ``grid_resolution``). Much faster, but the
    mesh is blocky and colorless (gray fallback). Pose matches the full pipeline.

    ``steps`` is the number of sparse-structure diffusion steps (default 20;
    the pipeline's native config uses 25).

    Returns a mesh dict in standard-pinhole camera frame.
    """
    start = time.time()

    rgb = np.asarray(image)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    elif rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)

    if not mask_u8.any():
        raise RuntimeError("mask has no non-zero pixels")
    if not np.any(np.isfinite(depth_map) & (depth_map > 0) & mask_u8.astype(bool)):
        raise RuntimeError("no valid depth pixels under the mask")

    result = generate_coarse(
        pipeline,
        image=rgb,
        depth=depth_map.astype(np.float32),
        mask=mask_u8,
        intrinsics=np.asarray(K, dtype=np.float64),
        grid_resolution=grid_resolution,
        sparse_structure_sampler_params={"steps": steps},
    )

    mesh_dict = _trimesh_to_dict(result.mesh)
    # Marching cubes over a binary occupancy grid can't produce the NaN vertices the
    # SLAT decoder can, but a non-finite pose would still poison every vertex here.
    mesh_dict = _drop_nonfinite(mesh_dict)
    mesh_dict = _decimate_mesh_dict(mesh_dict)
    elapsed = time.time() - start
    print(
        f"run_recgen_coarse done in {elapsed:.2f}s "
        f"(grid={grid_resolution}, steps={steps}, verts={len(mesh_dict['vertices'])}, faces={len(mesh_dict['faces'])})"
    )
    return mesh_dict


def _warmup() -> None:
    """Run one synthetic coarse request so the first real one isn't the slow one.

    Compiling the sparse-structure denoiser takes ~30s cold (~5s once
    TORCHINDUCTOR_CACHE_DIR is warm), and cuDNN/cuBLAS pick their algorithms on
    first use. Doing that here keeps it off the latency of a real request. The
    input just has to be a plausible RGB-D object; only shapes matter, and those
    are identical for every request.
    """
    start = time.time()
    h = w = 256
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    # A slightly tilted patch, so the unit-cube fit sees a non-degenerate extent.
    depth[80:176, 80:176] = 1.0 + np.linspace(0, 0.2, 96, dtype=np.float32)[:, None]
    mask[80:176, 80:176] = 1
    K = np.array([[200.0, 0.0, 128.0], [0.0, 200.0, 128.0], [0.0, 0.0, 1.0]])
    try:
        run_recgen_coarse(rgb, depth, mask, K, grid_resolution=32, steps=4)
        print(f"[recgen_fastapi] warmup done in {time.time() - start:.1f}s")
    except Exception as e:
        # Never block startup on the warmup — the compile still happens lazily.
        print(f"[recgen_fastapi] warmup failed ({type(e).__name__}: {e}); "
              f"first request will pay the compile cost")


# ── FastAPI app ──────────────────────────────────────────────────────────────

API_OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"), "orcd", "scratch", "api_outputs_recgen"
)

# Archival only: nothing in a request's behavior depends on these files. Off by
# default so a long-lived server doesn't accumulate a directory-per-request on
# NFS scratch; enable with --save-outputs (or RECGEN_SAVE_OUTPUTS=1) when you
# want to inspect what a caller actually sent.
SAVE_OUTPUTS = os.environ.get("RECGEN_SAVE_OUTPUTS", "0") != "0"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _warmup()
    yield


app = FastAPI(
    title="RecGen API",
    description="Single-view RGB-D mesh reconstruction server",
    lifespan=_lifespan,
)


@app.middleware("http")
async def _log_request_time(request: Request, call_next):
    """Log wall-clock per request, as the caller experiences it.

    Wider than the `run_recgen_*` lines, which cover only model + postprocess:
    this also includes multipart parsing of the uploads (which happens during
    dependency resolution, before the endpoint body runs), input decode, and
    pickling. The gap between the two numbers is the request overhead.

    Excludes writing the body to the socket, which happens after this returns —
    so on a slow client the caller still sees more than this reports.
    """
    if request.url.path == "/health":
        return await call_next(request)
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    print(
        f"[recgen_fastapi] {request.method} {request.url.path} "
        f"-> {response.status_code} in {elapsed:.2f}s"
    )
    return response


@app.get("/health")
async def health_check():
    # scene_cache: hits/decodes of the content-hashed frame decode. hits ~= 0 with
    # decodes climbing means callers are sending byte-varying frames (a re-encode
    # upstream) and the cache is dead weight — the client re-encoding a frame
    # non-deterministically would look exactly like this.
    return {"status": "healthy", "service": "recgen_server",
            "scene_cache": dict(SCENE_CACHE_STATS)}


@app.post("/reset_scene_cache/")
async def reset_scene_cache():
    """Drop the decoded-scene cache and zero its counters. A BENCH hook.

    The cache is keyed on a content hash, and a replayed session feeds
    byte-identical frames every run, so entries SURVIVE ACROSS RUNS while the OBM
    is rebuilt from scratch: the second arm to run a seed would start with the
    first arm's frames already decoded, and the advantage would land on whichever
    arm happened to go second. That is order-dependent and unreproducible — the
    within-experiment drift bench-pairing-drift warns about, which interleaving
    cannot cancel. Call this at run START so every run pays its own decodes.

    Within a run the cache is legitimate and stays on: one session really does
    request the same frame for many objects, which is the ~0.4s/repeat this exists
    to reclaim.
    """
    with _SCENE_CACHE_LOCK:
        n = len(_SCENE_CACHE)
        _SCENE_CACHE.clear()
        for k in SCENE_CACHE_STATS:
            SCENE_CACHE_STATS[k] = 0
    return {"cleared": n}


class _BadRequest(Exception):
    """Raised for input-shape mismatches so the endpoint returns HTTP 400."""


def _archive(experiment_id: str, files: dict) -> None:
    """Write `{name: bytes}` into this request's output dir. Archival only.

    Best-effort by design: this runs on requests that are otherwise fine, so a
    full or unwritable scratch mount must not turn a good completion into a 500.
    """
    if not SAVE_OUTPUTS:
        return
    try:
        experiment_dir = os.path.join(API_OUTPUT_DIR, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        for name, data in files.items():
            with open(os.path.join(experiment_dir, name), "wb") as f:
                f.write(data)
    except OSError as e:
        print(f"[recgen_fastapi] --save-outputs write failed for {experiment_id}: {e}")


# Decoding the ~3.4MB scene RGB-D costs ~0.4s and is a pure function of the
# uploaded bytes, but the single-object endpoints re-pay it on EVERY call and the
# client prices many objects against the same frame in a row: one measured
# off_frame_1 run made 46 scene-payload requests (complete + coarse) across only
# 15 distinct frames, so ~31 decodes (~12s/run) were recomputing a byte-identical
# result. Cache the decoded scene keyed on a content hash of the uploads.
#
# Entries are ~15MB (float32 depth + RGB), so the bound matters; 4 covers the
# repeat window (a featurize pass reuses a handful of frames) at ~60MB.
#
# INVARIANT: cached values are shared across requests and MUST NOT be mutated.
# Safe today because run_recgen_* copies before touching either — `np.asarray(image)`
# builds a fresh array and `depth_map.astype(np.float32)` copies (numpy's astype
# defaults to copy=True). A future in-place edit of `depth_map` would corrupt every
# later request for that frame, which would look like a model bug, not a cache bug.
#
# Is 4 enough? {hits, decodes} alone cannot say: an evicted-then-re-requested frame
# looks identical to a genuinely new one. `evictions` is the diagnostic —
# evictions == 0 means the bound never bound and a bigger cache buys nothing.
# Measured cold on off_frame_1: 37 requests / 16 distinct frames / 21 hits = the
# theoretical max (37-16), 0 evictions. Busier scenes (0708_3) are unmeasured.
_SCENE_CACHE_MAX = 4
_SCENE_CACHE: "OrderedDict[bytes, tuple]" = OrderedDict()
_SCENE_CACHE_LOCK = threading.Lock()
SCENE_CACHE_STATS = {"hits": 0, "decodes": 0, "evictions": 0}


def _scene_key(rgb_bytes, depth_bytes, intr_bytes, depth_name):
    """Content hash of the frame-level uploads. ~3ms on 3.4MB — worth it against a
    ~0.4s decode. `depth_name`'s extension is part of the key because _decode_depth
    dispatches on it (.npy metres vs .png millimetres), so the same bytes can decode
    two different ways."""
    h = hashlib.blake2b(digest_size=16)
    for blob in (rgb_bytes, depth_bytes, intr_bytes):
        h.update(len(blob).to_bytes(8, "little"))   # length-prefixed: no concat ambiguity
        h.update(blob)
    h.update(os.path.splitext(depth_name)[1].lower().encode())
    return h.digest()


def _decode_scene(rgb_bytes, depth_bytes, intr_bytes, depth_name):
    """Decode the per-frame (object-independent) inputs: (image, depth_map, K).

    Split out of :func:`_load_request_inputs` so /coarse_mesh_batch/ pays this
    once per frame instead of once per object — the scene RGB-D is ~3.4MB and its
    multipart parse + PIL decode was ~0.4s of every ~0.9s single call. Cached on a
    content hash, so repeat calls against the same frame skip it entirely.
    """
    key = _scene_key(rgb_bytes, depth_bytes, intr_bytes, depth_name)
    with _SCENE_CACHE_LOCK:
        hit = _SCENE_CACHE.get(key)
        if hit is not None:
            _SCENE_CACHE.move_to_end(key)
            SCENE_CACHE_STATS["hits"] += 1
            return hit
    # decode off-lock: two requests racing the same frame waste one decode, which
    # is cheaper than serialising every caller behind a ~0.4s hold
    image = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
    depth_map = _decode_depth(depth_bytes, depth_name)
    K = _decode_intrinsics(intr_bytes)
    if depth_map.shape[:2] != (image.size[1], image.size[0]):
        # not cached: a malformed frame must not occupy a slot, and it 400s anyway
        raise _BadRequest(f"depth shape {depth_map.shape[:2]} does not match rgb {(image.size[1], image.size[0])}")
    with _SCENE_CACHE_LOCK:
        SCENE_CACHE_STATS["decodes"] += 1
        _SCENE_CACHE[key] = (image, depth_map, K)
        _SCENE_CACHE.move_to_end(key)
        while len(_SCENE_CACHE) > _SCENE_CACHE_MAX:
            _SCENE_CACHE.popitem(last=False)
            SCENE_CACHE_STATS["evictions"] += 1
    return image, depth_map, K


def _decode_mask(mask_bytes, depth_map, what="mask"):
    """Decode one binary mask and check it against the frame. Raises _BadRequest."""
    mask_arr = np.asarray(Image.open(io.BytesIO(mask_bytes))) > 0
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    if mask_arr.shape[:2] != depth_map.shape[:2]:
        raise _BadRequest(f"{what} shape {mask_arr.shape[:2]} does not match depth {depth_map.shape[:2]}")
    return mask_arr


async def _load_request_inputs(experiment_id, rgb, depth, mask, intrinsics):
    """Decode and validate the four uploads.

    Returns (image, depth_map, mask_arr, K). Raises _BadRequest on shape mismatch.
    """
    rgb_bytes = await rgb.read()
    depth_bytes = await depth.read()
    mask_bytes = await mask.read()
    intr_bytes = await intrinsics.read()

    rgb_name = rgb.filename or "rgb.png"
    depth_name = depth.filename or "depth.png"
    mask_name = mask.filename or "mask.png"
    _archive(experiment_id, {
        "rgb" + os.path.splitext(rgb_name)[1]: rgb_bytes,
        "depth" + os.path.splitext(depth_name)[1]: depth_bytes,
        "mask" + os.path.splitext(mask_name)[1]: mask_bytes,
        "intrinsics.npy": intr_bytes,
    })

    image, depth_map, K = _decode_scene(rgb_bytes, depth_bytes, intr_bytes, depth_name)
    mask_arr = _decode_mask(mask_bytes, depth_map)
    return image, depth_map, mask_arr, K


def _pickle_and_respond(mesh_dict, experiment_id):
    """Pickle the mesh dict and return it as a downloadable response body."""
    payload = pickle.dumps(mesh_dict, protocol=pickle.HIGHEST_PROTOCOL)
    _archive(experiment_id, {"mesh.pkl": payload})
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="mesh_{experiment_id}.pkl"'},
    )


@app.post("/complete_mesh/")
async def complete_mesh_endpoint(
    rgb: UploadFile = File(..., description="Full UNMASKED scene RGB image (PNG)"),
    depth: UploadFile = File(..., description="Full UNMASKED scene depth (.png uint16 mm or .npy float32 m)"),
    mask: UploadFile = File(..., description="Per-object binary mask PNG (nonzero = object pixels)"),
    intrinsics: UploadFile = File(..., description="Camera intrinsics .npy (3x3 K or [fx,fy,cx,cy])"),
):
    """Run RecGen on one object given the full scene RGB-D plus a per-object mask.

    The mesh is returned in standard-pinhole camera frame (RecGen's
    `result.mesh`, equivalent to `posed_mesh.obj`). The pickle dict carries
    `vertices`, `faces`, `vertex_colors` — drop-in for sam3d_fastapi consumers
    modulo the frame difference (sam3d returns PyTorch3D camera frame).
    """
    experiment_id = f"{int(time.time() * 1000)}"
    try:
        image, depth_map, mask_arr, K = await _load_request_inputs(
            experiment_id, rgb, depth, mask, intrinsics
        )
        mesh_dict = run_recgen_single(image, depth_map, mask_arr, K)
        return _pickle_and_respond(mesh_dict, experiment_id)

    except _BadRequest as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except _DegenerateCompletion as e:
        # The request was fine; the model just had too little evidence for this
        # object to decode anything usable. Distinct from a 500 so callers can tell
        # "this object is hopeless, move on" from "the server is broken".
        return JSONResponse(
            content={"error": str(e), "reason": "degenerate_completion"}, status_code=422
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/coarse_mesh/")
async def coarse_mesh_endpoint(
    rgb: UploadFile = File(..., description="Full UNMASKED scene RGB image (PNG)"),
    depth: UploadFile = File(..., description="Full UNMASKED scene depth (.png uint16 mm or .npy float32 m)"),
    mask: UploadFile = File(..., description="Per-object binary mask PNG (nonzero = object pixels)"),
    intrinsics: UploadFile = File(..., description="Camera intrinsics .npy (3x3 K or [fx,fy,cx,cy])"),
    grid_resolution: int = 32,
    steps: int = 20,
):
    """Fast coarse variant: runs only RecGen's sparse-structure stage.

    Skips SLAT sampling + the mesh decoder and instead marching-cubes the coarse
    occupancy voxels (downsampled to `grid_resolution`, e.g. 16 or 32). Returns a
    blocky, **untextured** (gray) mesh in the same standard-pinhole camera frame
    as /complete_mesh/, with the same pose. Much faster; use when geometry/pose
    matter more than appearance.

    `steps` sets the number of sparse-structure diffusion steps (default 20; the
    pipeline's native config uses 25). Fewer steps is faster but coarser.
    """
    experiment_id = f"{int(time.time() * 1000)}"
    try:
        image, depth_map, mask_arr, K = await _load_request_inputs(
            experiment_id, rgb, depth, mask, intrinsics
        )
        mesh_dict = run_recgen_coarse(image, depth_map, mask_arr, K, grid_resolution=grid_resolution, steps=steps)
        return _pickle_and_respond(mesh_dict, experiment_id)

    except _BadRequest as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except _DegenerateCompletion as e:
        return JSONResponse(
            content={"error": str(e), "reason": "degenerate_completion"}, status_code=422
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/coarse_mesh_batch/")
async def coarse_mesh_batch_endpoint(
    rgb: UploadFile = File(..., description="Full UNMASKED scene RGB image (PNG), sent ONCE"),
    depth: UploadFile = File(..., description="Full UNMASKED scene depth (.png uint16 mm or .npy float32 m), sent ONCE"),
    masks: list[UploadFile] = File(..., description="N per-object binary mask PNGs for the SAME frame"),
    intrinsics: UploadFile = File(..., description="Camera intrinsics .npy (3x3 K or [fx,fy,cx,cy])"),
    grid_resolution: int = 32,
    steps: int = 20,
):
    """N objects from ONE frame in one request: /coarse_mesh/ batched over masks.

    Motivation (coarse_latency_tax_0714_night §5): the policy's first featurize
    pass prices ~19-21 candidates against a single freshly-ingested frame, and
    per-call the shared ~3.4MB scene payload plus its multipart parse and PIL
    decode (~0.4s of a ~0.9s call) was paid once PER OBJECT. Here it is paid once
    per frame, so N objects cost ~upload + parse + N x compute instead of
    N x (upload + parse + compute).

    Returns a pickled ``list[dict]``, one entry per input mask, **in input
    order**:
        {"index": i, "name": <mask filename>, "ok": True,  "mesh": {vertices, faces, vertex_colors}}
        {"index": i, "name": <mask filename>, "ok": False, "error": str, "reason": str}

    **Per-object failures do not fail the batch.** A degenerate object (the
    /coarse_mesh/ 422 case — concave_heavy_1's green_koosh_ball 422s on ~13% of
    mints) comes back as one ``ok=False`` entry with reason
    ``degenerate_completion`` while its siblings return meshes; an all-degenerate
    batch is still HTTP 200 with every entry ``ok=False``. Only frame-level
    problems (rgb/depth shape mismatch, undecodable intrinsics) 400 the request,
    because those invalidate every object. Callers must check ``ok`` per entry.

    Meshes are identical to what /coarse_mesh/ returns for the same (frame, mask):
    same standard-pinhole camera frame, same pose, same grid_resolution/steps
    semantics. This endpoint only amortizes the per-request scene work — the
    diffusion still runs once per object, sequentially. GPU-batching the
    sparse-structure stage across objects is the larger prize and is NOT done here.
    """
    experiment_id = f"{int(time.time() * 1000)}"
    start = time.time()
    try:
        rgb_bytes = await rgb.read()
        depth_bytes = await depth.read()
        intr_bytes = await intrinsics.read()
        mask_blobs = [(m.filename or f"mask_{i}.png", await m.read())
                      for i, m in enumerate(masks)]

        rgb_name = rgb.filename or "rgb.png"
        depth_name = depth.filename or "depth.png"
        _archive(experiment_id, {
            "rgb" + os.path.splitext(rgb_name)[1]: rgb_bytes,
            "depth" + os.path.splitext(depth_name)[1]: depth_bytes,
            "intrinsics.npy": intr_bytes,
            **{f"mask_{i}" + os.path.splitext(name)[1]: blob
               for i, (name, blob) in enumerate(mask_blobs)},
        })

        # Frame-level: decode once, and a failure here 400s the whole batch.
        image, depth_map, K = _decode_scene(rgb_bytes, depth_bytes, intr_bytes, depth_name)

        results = []
        for i, (name, blob) in enumerate(mask_blobs):
            # Object-level: isolate every failure to its own entry.
            try:
                mask_arr = _decode_mask(blob, depth_map, what=f"mask[{i}] ({name})")
                mesh_dict = run_recgen_coarse(image, depth_map, mask_arr, K,
                                              grid_resolution=grid_resolution, steps=steps)
                results.append({"index": i, "name": name, "ok": True, "mesh": mesh_dict})
            except _DegenerateCompletion as e:
                results.append({"index": i, "name": name, "ok": False,
                                "error": str(e), "reason": "degenerate_completion"})
            except _BadRequest as e:
                results.append({"index": i, "name": name, "ok": False,
                                "error": str(e), "reason": "bad_request"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                results.append({"index": i, "name": name, "ok": False,
                                "error": f"{type(e).__name__}: {e}", "reason": "error"})

        n_ok = sum(r["ok"] for r in results)
        print(f"coarse_mesh_batch done in {time.time() - start:.2f}s "
              f"({n_ok}/{len(results)} ok, grid={grid_resolution}, steps={steps})")
        return _pickle_and_respond(results, experiment_id)

    except _BadRequest as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Archive each request's uploads and resulting mesh.pkl under "
             f"{API_OUTPUT_DIR}/<request-id>/. Off by default; affects nothing "
             "about the response.",
    )
    args = parser.parse_args()
    if args.save_outputs:
        SAVE_OUTPUTS = True
        os.makedirs(API_OUTPUT_DIR, exist_ok=True)
        print(f"[recgen_fastapi] --save-outputs: archiving to {API_OUTPUT_DIR}")

    # Pass the app object, not "recgen_fastapi:app": the import-string form makes
    # uvicorn import this module a second time (it is __main__ here, so the import
    # doesn't hit sys.modules), which builds and uploads a whole second copy of the
    # pipeline. Only reload/workers need the import string, and both are off.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8040,
        reload=False,
    )
