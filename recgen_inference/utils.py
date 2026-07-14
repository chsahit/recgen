"""Small helpers shared by build_recgen, inference, and user code.

This module holds:
    - parse_pose: turn the model's pose tensor into a 4x4 matrix + scipy dict
    - mesh_from_result: pull a trimesh (with vertex colors) out of a pipeline MeshExtractResult
    - load_intrinsics / read_pose_file: ingestion helpers for camera parameters
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import json
import numpy as np
import trimesh

from recgen_inference.recgen_modules.utils.pose_utils import parse_pose_output


POSE_DIM_TO_REPRESENTATION = {
    8: "quaternion_translation_scale",
    10: "6d_translation_scale",
    13: "9d_translation_scale",
}


def parse_pose(
    outputs: Dict[str, Any],
    pose_representation: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """Extract a 4x4 transformation matrix from raw pipeline outputs.

    Args:
        outputs: Raw dict from `pipeline.run_pointmap*` (must contain `outputs['pose']`).
        pose_representation: One of the string keys in POSE_DIM_TO_REPRESENTATION.
            If None, infer from the tensor's last dimension.

    Returns:
        T: (4, 4) float64 transformation matrix (rotation * scale, then translation).
        parsed: dict with keys 'quaternion', 'translation', 'scale', 'rotation_matrix'.
        pose_representation: the resolved representation string.
    """
    pose_tensor = outputs["pose"][0]

    if pose_representation is None:
        dim = pose_tensor.shape[0]
        pose_representation = POSE_DIM_TO_REPRESENTATION.get(dim)
        if pose_representation is None:
            raise ValueError(f"Unexpected pose tensor dimension: {dim}")

    parsed = parse_pose_output(pose_tensor, pose_representation)

    T = np.eye(4)
    T[:3, :3] = parsed["rotation_matrix"] * parsed["scale"]
    T[:3, 3] = parsed["translation"]
    return T, parsed, pose_representation


def mesh_from_result(mesh_result: Any) -> trimesh.Trimesh:
    """Build a trimesh (with vertex colors if available) from a MeshExtractResult.

    The pipeline's `MeshExtractResult.vertex_attrs` stores `[:, :3]` as RGB in [0, 1]
    and `[:, 3:]` as normals. Vertex colors are attached as RGBA uint8.
    """
    verts = mesh_result.vertices.detach().cpu().numpy()
    faces = mesh_result.faces.detach().cpu().numpy()

    vertex_colors = None
    if mesh_result.vertex_attrs is not None and mesh_result.vertex_attrs.shape[1] >= 3:
        rgb = mesh_result.vertex_attrs[:, :3].detach().cpu().numpy()
        rgb = np.clip(rgb, 0.0, 1.0)
        alpha = np.ones((rgb.shape[0], 1), dtype=rgb.dtype)
        vertex_colors = (np.concatenate([rgb, alpha], axis=1) * 255).astype(np.uint8)

    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors)


def coarse_mesh_from_coords(
    coords: Any,
    res: int = 64,
    grid_resolution: int = 32,
) -> trimesh.Trimesh:
    """Marching-cubes mesh from the coarse sparse-structure occupancy voxels.

    The sparse-structure stage yields occupancy voxel indices in ``[0, res)`` (the
    SLAT resolution). This bins them into a dense ``grid_resolution^3`` volume,
    downsampling ``res -> grid_resolution`` for a coarser/faster surface, and runs
    marching cubes. Vertices are mapped into the same normalized cube frame the
    SLAT mesh uses (``index / grid_resolution - 0.5``), so the same pose /
    ``cam2ncam`` transform places the mesh in the camera frame.

    Args:
        coords: (N, 3) or (N, 4) int voxel indices. A 4th leading column (batch)
            is dropped. Accepts a torch tensor or numpy array.
        res: Full occupancy grid resolution the coords index (SLAT resolution).
        grid_resolution: Target volume resolution for marching cubes. Smaller =
            coarser and faster (e.g. 16 or 32).

    Returns:
        An (untextured) ``trimesh.Trimesh`` in the normalized cube ``[-0.5, 0.5]^3``.
    """
    from skimage.measure import marching_cubes

    if hasattr(coords, "detach"):
        coords = coords.detach().cpu().numpy()
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"coords must be (N, 3) or (N, 4), got {coords.shape}")
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    if coords.shape[0] == 0:
        raise ValueError("coarse occupancy is empty (no voxels)")

    D = int(grid_resolution)
    idx = np.clip(
        (coords.astype(np.float64) * D / float(res)).astype(np.int64), 0, D - 1
    )
    vol = np.zeros((D, D, D), dtype=np.float32)
    vol[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    # Pad with a zero border so the surface closes at the volume boundary.
    vol = np.pad(vol, 1, mode="constant", constant_values=0.0)

    verts, faces, normals, _ = marching_cubes(vol, level=0.5)
    verts = verts - 1.0  # undo the pad offset
    verts = verts / D - 0.5  # grid index -> normalized cube [-0.5, 0.5]
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)


def intrinsics_from_params(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a (3, 3) float32 intrinsics matrix from individual focal / principal-point values."""
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def load_intrinsics(path: Union[str, Path]) -> np.ndarray:
    """Load a (3, 3) intrinsics matrix from a JSON file.

    The JSON must contain the four keys `fx`, `fy`, `cx`, `cy`.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return intrinsics_from_params(data["fx"], data["fy"], data["cx"], data["cy"])


def read_pose_file(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Read a pose file containing 4x4 extrinsics, a blank line, and 3x3 intrinsics.

    Returns:
        extrinsics: (4, 4) float32
        intrinsics: (3, 3) float32
    """
    lines = Path(path).read_text().splitlines()
    extrinsics_rows = [list(map(float, lines[i].split())) for i in range(4)]
    intrinsics_rows = [list(map(float, lines[5 + i].split())) for i in range(3)]
    return (
        np.array(extrinsics_rows, dtype=np.float32),
        np.array(intrinsics_rows, dtype=np.float32),
    )
