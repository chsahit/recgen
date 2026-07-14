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

import pickle
import time

import fast_simplification
import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sklearn.neighbors import KDTree

from recgen_inference import build_recgen, generate, generate_coarse


# ── Model loading (module-level, done once) ───────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_NAME = os.environ.get("RECGEN_PIPELINE", "recgen_base.multiview_stereo")
pipeline = build_recgen.build(PIPELINE_NAME)


# ── Input loaders (match sam3d_fastapi semantics) ─────────────────────────────

def _load_depth_file(path: str) -> np.ndarray:
    """Load a depth map as float32 meters with 0 where invalid.

    Accepts .npy (float32 meters) or .png (uint16 millimeters following the
    existing capture-zip convention). Non-finite pixels are zeroed out.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        depth = np.load(path).astype(np.float32)
    else:
        arr = np.array(Image.open(path))
        if arr.dtype == np.uint16:
            depth = arr.astype(np.float32) / 1000.0
        else:
            depth = arr.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def _load_intrinsics_file(path: str) -> np.ndarray:
    """Load intrinsics as a 3x3 K matrix. Accepts 3x3 K or 4-vector [fx,fy,cx,cy]."""
    K = np.load(path)
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

    mesh_dict = _trimesh_to_dict(result.mesh)
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
    mesh_dict = _decimate_mesh_dict(mesh_dict)
    elapsed = time.time() - start
    print(
        f"run_recgen_coarse done in {elapsed:.2f}s "
        f"(grid={grid_resolution}, steps={steps}, verts={len(mesh_dict['vertices'])}, faces={len(mesh_dict['faces'])})"
    )
    return mesh_dict


# ── FastAPI app ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
API_OUTPUT_DIR = os.path.join(os.environ["HOME"], "orcd", "scratch", "api_outputs_recgen")

app = FastAPI(
    title="RecGen API",
    description="Single-view RGB-D mesh reconstruction server",
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "recgen_server"}


class _BadRequest(Exception):
    """Raised for input-shape mismatches so the endpoint returns HTTP 400."""


async def _load_request_inputs(experiment_dir, rgb, depth, mask, intrinsics):
    """Save the four uploads to `experiment_dir` and load/validate them.

    Returns (image, depth_map, mask_arr, K). Raises _BadRequest on shape mismatch.
    """
    rgb_path = os.path.join(experiment_dir, "rgb" + os.path.splitext(rgb.filename or "rgb.png")[1])
    depth_path = os.path.join(experiment_dir, "depth" + os.path.splitext(depth.filename or "depth.png")[1])
    mask_path = os.path.join(experiment_dir, "mask" + os.path.splitext(mask.filename or "mask.png")[1])
    intr_path = os.path.join(experiment_dir, "intrinsics.npy")
    with open(rgb_path, "wb") as f:
        f.write(await rgb.read())
    with open(depth_path, "wb") as f:
        f.write(await depth.read())
    with open(mask_path, "wb") as f:
        f.write(await mask.read())
    with open(intr_path, "wb") as f:
        f.write(await intrinsics.read())

    image = Image.open(rgb_path).convert("RGB")
    depth_map = _load_depth_file(depth_path)
    K = _load_intrinsics_file(intr_path)
    mask_arr = np.asarray(Image.open(mask_path)) > 0
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]

    if depth_map.shape[:2] != (image.size[1], image.size[0]):
        raise _BadRequest(f"depth shape {depth_map.shape[:2]} does not match rgb {(image.size[1], image.size[0])}")
    if mask_arr.shape[:2] != depth_map.shape[:2]:
        raise _BadRequest(f"mask shape {mask_arr.shape[:2]} does not match depth {depth_map.shape[:2]}")

    return image, depth_map, mask_arr, K


def _pickle_and_respond(mesh_dict, experiment_dir, experiment_id):
    """Pickle the mesh dict and return it as a downloadable FileResponse."""
    out_path = os.path.join(experiment_dir, "mesh.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(mesh_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return FileResponse(
        path=out_path,
        filename=f"mesh_{experiment_id}.pkl",
        media_type="application/octet-stream",
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
    experiment_dir = os.path.join(API_OUTPUT_DIR, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    try:
        image, depth_map, mask_arr, K = await _load_request_inputs(
            experiment_dir, rgb, depth, mask, intrinsics
        )
        mesh_dict = run_recgen_single(image, depth_map, mask_arr, K)
        return _pickle_and_respond(mesh_dict, experiment_dir, experiment_id)

    except _BadRequest as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
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
    experiment_dir = os.path.join(API_OUTPUT_DIR, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    try:
        image, depth_map, mask_arr, K = await _load_request_inputs(
            experiment_dir, rgb, depth, mask, intrinsics
        )
        mesh_dict = run_recgen_coarse(image, depth_map, mask_arr, K, grid_resolution=grid_resolution, steps=steps)
        return _pickle_and_respond(mesh_dict, experiment_dir, experiment_id)

    except _BadRequest as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    os.makedirs(API_OUTPUT_DIR, exist_ok=True)
    uvicorn.run(
        "recgen_fastapi:app",
        host="0.0.0.0",
        port=8040,
        reload=False,
    )
