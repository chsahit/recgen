"""Inference entry points: generate (single-view) and generate_multiview."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from recgen_inference._result import RecGenResult, build_result
from recgen_inference.preprocessing import (
    apply_mask_erosion,
    crop_to_bounding_box,
    normalize_depth,
    pointmap_from_depth,
    preprocess_view,
)
from recgen_inference.utils import coarse_mesh_from_coords, mesh_from_result, parse_pose


def _preprocess_single(
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    quantile_drop_threshold: float,
    clamp_range: Tuple[float, float],
    mask_erosion_enabled: bool,
    mask_erosion_params: Optional[Dict[str, int]],
) -> Dict[str, Any]:
    """Preprocess one view into tensors the pipeline can consume."""
    depth = normalize_depth(depth)

    mask_eroded, _ = apply_mask_erosion(
        mask.copy(), enabled=mask_erosion_enabled, params=mask_erosion_params
    )
    if mask_eroded.dtype == bool:
        mask_eroded = mask_eroded.astype(np.uint8)
    if mask_eroded.max() == 1:
        mask_eroded = mask_eroded * 255

    depth_masked = depth.copy()
    depth_masked[mask_eroded == 0] = 0.0
    valid = depth_masked > 0

    pil_image = Image.fromarray(image)
    pointmap_scene = pointmap_from_depth(depth_masked, intrinsics)

    pointmap_tensor, resized_image, resized_mask, cam2ncam = crop_to_bounding_box(
        pointmap_scene,
        pil_image,
        Image.fromarray(mask_eroded),
        valid,
        quantile_drop_threshold=quantile_drop_threshold,
        clamp_range=clamp_range,
    )

    return {
        "image": resized_image,
        "pointmap": pointmap_tensor,
        "mask": resized_mask,
        "cam2ncam": cam2ncam,
    }


def _build_result_from_outputs(
    outputs: Dict[str, Any],
    cam2ncam: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    raw_trimesh: Optional[Any] = None,
) -> RecGenResult:
    """Shared postprocessing path: parse pose, build mesh, transform into camera frame.

    If ``raw_trimesh`` is provided it is used directly (e.g. the coarse
    marching-cubes mesh); otherwise the mesh is decoded from ``outputs['mesh']``.
    """
    pose_matrix, parsed_pose, pose_representation = parse_pose(outputs)

    if raw_trimesh is None:
        raw_trimesh = mesh_from_result(outputs["mesh"][0])

    final_mesh = raw_trimesh.copy()
    final_mesh.apply_transform(pose_matrix)
    final_mesh.apply_translation(-cam2ncam[:3, 3])
    final_mesh.apply_scale(1.0 / cam2ncam[0, 0])

    return build_result(
        mesh=final_mesh,
        raw_mesh=raw_trimesh,
        pose_matrix=pose_matrix,
        parsed_pose=parsed_pose,
        cam2ncam=cam2ncam,
        pose_representation=pose_representation,
        outputs=outputs,
        rgb=rgb,
        intrinsics=intrinsics,
    )


def generate(
    pipeline: Any,
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    seed: int = 1,
    quantile_drop_threshold: float = 0.05,
    clamp_range: Tuple[float, float] = (-2.0, 3.0),
    mask_erosion_enabled: bool = True,
    mask_erosion_params: Optional[Dict[str, int]] = None,
    use_pointmap: bool = True,
    formats: List[str] = ["mesh", "gaussian", "radiance_field"],
) -> RecGenResult:
    """Generate a 3D mesh from a single RGB-D view.

    Args:
        pipeline: Pipeline from :func:`recgen_inference.build_recgen.build`.
        image: (H, W, 3) uint8 RGB numpy array.
        depth: (H, W) depth map. Accepts uint16 mm or float32 m — the unit is
            auto-detected by :func:`preprocessing.normalize_depth`.
        mask: (H, W) object mask. Non-zero pixels mark the object.
        intrinsics: (3, 3) camera intrinsic matrix.
        seed: Random seed.
        quantile_drop_threshold: Fraction of points to drop from each end when
            computing the robust unit-cube fit.
        clamp_range: Range into which the normalised pointmap is clamped.
        mask_erosion_enabled: Whether to erode the mask before preprocessing.
        mask_erosion_params: Optional ``{"kernel_size": int, "iterations": int}`` override.
        use_pointmap: If False, the pipeline runs without the pointmap branch.
        formats: Which SLAT decoders to run. Pass ``["mesh"]`` to skip the
            gaussian / radiance-field decoders when only the mesh is needed.
    """
    proc = _preprocess_single(
        image,
        depth,
        mask,
        intrinsics,
        quantile_drop_threshold=quantile_drop_threshold,
        clamp_range=clamp_range,
        mask_erosion_enabled=mask_erosion_enabled,
        mask_erosion_params=mask_erosion_params,
    )

    outputs = pipeline.run_pointmap(
        proc["image"],
        pointmap=proc["pointmap"] if use_pointmap else None,
        mask=proc["mask"],
        seed=seed,
        formats=formats,
    )

    return _build_result_from_outputs(outputs, proc["cam2ncam"], rgb=image, intrinsics=intrinsics)


def generate_coarse(
    pipeline: Any,
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    seed: int = 1,
    quantile_drop_threshold: float = 0.05,
    clamp_range: Tuple[float, float] = (-2.0, 3.0),
    mask_erosion_enabled: bool = True,
    mask_erosion_params: Optional[Dict[str, int]] = None,
    use_pointmap: bool = True,
    grid_resolution: int = 32,
    sparse_structure_sampler_params: Optional[Dict[str, Any]] = None,
) -> RecGenResult:
    """Generate a *coarse* mesh from a single RGB-D view (sparse-structure stage only).

    This runs only the first pipeline stage — sparse-structure + pose sampling —
    and turns the occupancy voxels into an untextured mesh via marching cubes. It
    skips the expensive SLAT sampling + mesh decoder, so it is substantially
    faster than :func:`generate`, at the cost of a blocky, colorless mesh. Pose is
    identical to the full pipeline (SLAT does not refine pose).

    Args:
        grid_resolution: Voxel resolution for the marching-cubes volume. Smaller =
            coarser and faster (e.g. 16 or 32). The occupancy is downsampled from
            the model's native resolution to this before meshing.
        sparse_structure_sampler_params: Optional overrides for the sparse-structure
            sampler (e.g. ``{"steps": 12}`` to trade quality for speed).

    Returns a :class:`RecGenResult` whose ``mesh`` has gray fallback colors and an
    empty ``_outputs`` (no gaussian / radiance field).
    """
    proc = _preprocess_single(
        image,
        depth,
        mask,
        intrinsics,
        quantile_drop_threshold=quantile_drop_threshold,
        clamp_range=clamp_range,
        mask_erosion_enabled=mask_erosion_enabled,
        mask_erosion_params=mask_erosion_params,
    )

    coarse = pipeline.run_pointmap_coarse(
        proc["image"],
        pointmap=proc["pointmap"] if use_pointmap else None,
        mask=proc["mask"],
        seed=seed,
        sparse_structure_sampler_params=sparse_structure_sampler_params or {},
    )

    res = getattr(pipeline.models["slat_flow_model"], "resolution", 64)
    raw_trimesh = coarse_mesh_from_coords(
        coarse["coords"], res=res, grid_resolution=grid_resolution
    )

    outputs = {"pose": coarse["pose"]}
    return _build_result_from_outputs(
        outputs, proc["cam2ncam"], rgb=image, intrinsics=intrinsics, raw_trimesh=raw_trimesh
    )


def generate_multiview(
    pipeline: Any,
    anchor_view: Dict[str, Any],
    second_views: List[Dict[str, Any]],
    *,
    seed: int = 1,
    quantile_drop_threshold: float = 0.05,
    clamp_range: Tuple[float, float] = (-2.0, 3.0),
    mask_erosion_enabled: bool = True,
    mask_erosion_params: Optional[Dict[str, int]] = None,
    use_pointmap: bool = True,
) -> RecGenResult:
    """Generate a 3D mesh from an anchor view + N supporting views.

    Each view dict must contain:
        - ``rgb``: (H, W, 3) uint8
        - ``depth``: (H, W) depth (uint16 mm or float32 m — auto-detected)
        - ``mask``: (H, W) object mask
        - ``camera_intrinsics``: (3, 3)

    The result is expressed in the anchor view's camera frame.
    """
    if len(second_views) < 1:
        raise ValueError("generate_multiview needs at least one second view")

    proc1 = preprocess_view(
        anchor_view["rgb"],
        normalize_depth(anchor_view["depth"]),
        anchor_view["mask"],
        anchor_view["camera_intrinsics"],
        quantile_drop_threshold=quantile_drop_threshold,
        clamp_range=clamp_range,
        mask_erosion_enabled=mask_erosion_enabled,
        mask_erosion_params=mask_erosion_params,
    )
    procs2 = [
        preprocess_view(
            v["rgb"],
            normalize_depth(v["depth"]),
            v["mask"],
            v["camera_intrinsics"],
            quantile_drop_threshold=quantile_drop_threshold,
            clamp_range=clamp_range,
            mask_erosion_enabled=mask_erosion_enabled,
            mask_erosion_params=mask_erosion_params,
        )
        for v in second_views
    ]

    all_procs = [proc1] + procs2
    outputs = pipeline.run_pointmap_multiview(
        images=[p["image"] for p in all_procs],
        pointmaps=[p["pointmap"] for p in all_procs] if use_pointmap else None,
        masks=[p["mask"] for p in all_procs],
        seed=seed,
    )

    return _build_result_from_outputs(
        outputs,
        proc1["cam2ncam"],
        rgb=anchor_view["rgb"],
        intrinsics=anchor_view["camera_intrinsics"],
    )
