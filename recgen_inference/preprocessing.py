"""Input preprocessing: depth→pointmap, mask erosion, normalization, cropping."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a declared dep, this is belt-and-braces
    cv2 = None


# ---------------------------------------------------------------------------
# Depth unit normalization
# ---------------------------------------------------------------------------

def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Return depth in metres regardless of input dtype or unit.

    Accepts:
      - uint16 PNG depth in millimetres (common Kinect / RealSense format).
      - float32 already in metres.
      - float32 in millimetres (auto-detected via magnitude).

    Heuristic: if the input is an integer dtype, or if a float array has a
    maximum > 30 (effectively impossible for metre-scale indoor scenes), we
    treat it as millimetres and divide by 1000.
    """
    depth = np.asarray(depth)
    if np.issubdtype(depth.dtype, np.integer):
        return depth.astype(np.float32) / 1000.0
    depth = depth.astype(np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size > 0 and finite.max() > 30.0:
        return depth / 1000.0
    return depth


# ---------------------------------------------------------------------------
# Mask erosion
# ---------------------------------------------------------------------------

def apply_mask_erosion(
    mask: np.ndarray,
    enabled: bool = True,
    params: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply binary erosion to an object mask.

    Args:
        mask: Binary mask (numpy array, uint8 or bool).
        enabled: Whether to apply erosion.
        params: Dict with 'kernel_size' and 'iterations' (defaults: 5, 1).

    Returns:
        eroded_mask: uint8 mask (0 or 255).
        erosion_info: Dict with statistics.
    """
    if params is None:
        params = {}
    kernel_size = params.get('kernel_size', 5)
    iterations = params.get('iterations', 1)

    mask_bool = mask.astype(bool)
    pixels_before = int(np.sum(mask_bool))

    if enabled and iterations > 0:
        if cv2 is not None:
            # ~40x faster than scipy on a full frame. borderValue=0 reproduces
            # scipy's border_value=0, which erodes masks that touch the frame edge.
            eroded_mask = cv2.erode(
                mask_bool.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8),
                iterations=iterations, borderType=cv2.BORDER_CONSTANT, borderValue=0,
            ) * 255
        else:
            structure = np.ones((kernel_size, kernel_size), dtype=bool)
            eroded_bool = binary_erosion(mask_bool, structure=structure, iterations=iterations)
            eroded_mask = eroded_bool.astype(np.uint8) * 255
    else:
        eroded_mask = mask_bool.astype(np.uint8) * 255

    pixels_after = int(np.sum(eroded_mask > 0))
    erosion_info = {
        'enabled': enabled,
        'kernel_size': kernel_size,
        'iterations': iterations,
        'pixels_before': pixels_before,
        'pixels_after': pixels_after,
        'pixels_removed': pixels_before - pixels_after,
        'percent_removed': 100.0 * (pixels_before - pixels_after) / max(pixels_before, 1),
    }
    return eroded_mask, erosion_info




# ---------------------------------------------------------------------------
# Depth → pointmap
# ---------------------------------------------------------------------------

def pointmap_from_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project a depth image to a 3D pointmap.

    Args:
        depth: (H, W) depth in metres.
        intrinsics: (3, 3) camera intrinsic matrix.

    Returns:
        (H, W, 3) numpy array of 3D points.
    """
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    z = depth
    x = (i - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (j - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.stack((x, y, z), axis=-1)


# ---------------------------------------------------------------------------
# Unit-cube fitting
# ---------------------------------------------------------------------------


def fit_unit_cube_median_quantile(X, quantile_drop_threshold=0.025):
    """Fit point cloud into [0,1]^D using median + quantile (robust to outliers)."""
    X = np.asarray(X)
    shift = np.nanmedian(X, axis=0)
    shifted = X - shift
    norms = np.linalg.norm(shifted, axis=1)

    lower_q = np.nanquantile(norms, quantile_drop_threshold)
    upper_q = np.nanquantile(norms, 1.0 - quantile_drop_threshold)
    diameter = (upper_q - lower_q) * 2.0

    if diameter == 0:
        s = 1.0
        t = np.full(X.shape[1], 0.5) - shift
    else:
        s = 1.0 / diameter
        t = np.full(X.shape[1], 0.5) - shift * s

    T = np.eye(4)
    T[0, 0] = T[1, 1] = T[2, 2] = s
    T[:3, 3] = t
    return T, s, t


# ---------------------------------------------------------------------------
# Crop and normalise
# ---------------------------------------------------------------------------

def _aug_bbox_from_mask(mask_arr, image_size=518, aug_size_ratio=1.2, max_increase_ratio=3.0):
    """Object bounding box, squared off and padded by `aug_size_ratio`.

    `mask_arr` is a (H, W) array; the returned box may extend outside the frame.
    """
    nz = mask_arr.nonzero()
    if nz[0].size == 0 or nz[1].size == 0:
        height, width = mask_arr.shape[:2]
        bbox = [0, 0, width, height]
    else:
        bbox = [nz[1].min(), nz[0].min(), nz[1].max(), nz[0].max()]

    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    min_hsize = (image_size / max_increase_ratio) / 2
    aug_hsize = max(hsize * aug_size_ratio, min_hsize)

    return [int(center[0] - aug_hsize), int(center[1] - aug_hsize),
            int(center[0] + aug_hsize), int(center[1] + aug_hsize)]


def crop_to_bounding_box_cropped(depth_masked, intrinsics, image, mask_eroded, valid,
                                 image_size=518, aug_size_ratio=1.2, max_increase_ratio=3.0,
                                 quantile_drop_threshold=0.025,
                                 clamp_range=(-2.0, 3.0)):
    """Same outputs as :func:`crop_to_bounding_box`, without the full-frame pointmap.

    :func:`crop_to_bounding_box` back-projects and normalises every pixel in the
    frame, then throws away everything outside the object's augmented bbox — on a
    1080x1920 input that is ~30x more work than the crop needs. This computes the
    bbox first and back-projects only the object pixels inside it, which is exact:
    the bbox always contains the mask, and the non-mask pixels it keeps are zeroed
    anyway.

    Args:
        depth_masked: (H, W) float depth in metres, already zeroed outside the mask.
        intrinsics: (3, 3) camera intrinsic matrix.
        image: full-frame PIL image.
        mask_eroded: (H, W) uint8 mask (0 or 255).
        valid: (H, W) bool, True where depth is usable.

    Returns:
        pointmap (3, H, W) tensor, resized_image PIL, resized_mask PIL, cam2ncam (4,4) ndarray
    """
    H, W = depth_masked.shape[:2]
    aug_bbox = _aug_bbox_from_mask(mask_eroded, image_size, aug_size_ratio, max_increase_ratio)
    left, upper, right, lower = aug_bbox
    out_h, out_w = lower - upper, right - left

    image_resized = image.crop(tuple(aug_bbox)).resize((image_size, image_size), Image.Resampling.LANCZOS)
    mask_resized = Image.fromarray(mask_eroded).crop(tuple(aug_bbox)).resize(
        (image_size, image_size), Image.Resampling.NEAREST)

    # Intersect the (possibly out-of-frame) bbox with the image; the rest stays zero.
    src_top, src_bottom = max(upper, 0), min(lower, H)
    src_left, src_right = max(left, 0), min(right, W)
    dst_top, dst_left = src_top - upper, src_left - left

    sel = (mask_eroded[src_top:src_bottom, src_left:src_right] > 0) & valid[src_top:src_bottom, src_left:src_right]
    ys, xs = np.nonzero(sel)

    # Back-project just the selected pixels. float64 to match the full-frame path,
    # which builds the pointmap in float64 and only narrows to float32 afterwards.
    z = depth_masked[src_top:src_bottom, src_left:src_right][ys, xs].astype(np.float64)
    X = np.empty((ys.size, 3), dtype=np.float64)
    X[:, 0] = (xs + src_left - intrinsics[0, 2]) * z / intrinsics[0, 0]
    X[:, 1] = (ys + src_top - intrinsics[1, 2]) * z / intrinsics[1, 1]
    X[:, 2] = z
    X = X.astype(np.float32)

    cam2ncam, s, _ = fit_unit_cube_median_quantile(X, quantile_drop_threshold)

    X_ncam = (X @ cam2ncam[:3, :3].T + cam2ncam[:3, 3]).astype(np.float32)

    pointmap_ncam_crop = np.zeros((3, out_h, out_w), dtype=np.float32)
    pointmap_ncam_crop[:, ys + dst_top, xs + dst_left] = X_ncam.T

    pointmap_ncam = F.interpolate(
        torch.from_numpy(pointmap_ncam_crop)[None], size=(image_size, image_size), mode='nearest'
    )[0].contiguous()

    pointmap = pointmap_ncam.clone()
    if clamp_range is not None:
        pointmap = pointmap.clamp(*clamp_range)

    return pointmap, image_resized, mask_resized, cam2ncam


def crop_to_bounding_box(pointmap_scene, image, mask, valid,
                         image_size=518, aug_size_ratio=1.2, max_increase_ratio=3.0,
                         quantile_drop_threshold=0.025,
                         clamp_range=(-2.0, 3.0)):
    """Crop to object bounding box, normalise pointmap to unit cube.

    Returns:
        pointmap (3, H, W) tensor, resized_image PIL, resized_mask PIL, cam2ncam (4,4) ndarray
    """
    aug_bbox = _aug_bbox_from_mask(
        np.array(mask), image_size, aug_size_ratio, max_increase_ratio)

    image_resized = image.crop(aug_bbox).resize((image_size, image_size), Image.Resampling.LANCZOS)
    mask_resized = mask.crop(aug_bbox).resize((image_size, image_size), Image.Resampling.NEAREST)

    mask_tensor = torch.from_numpy(np.array(mask)).bool() & torch.from_numpy(valid).bool()
    pointmap_3d = torch.from_numpy(pointmap_scene).float().permute(2, 0, 1)
    X = pointmap_3d[:, mask_tensor].permute(1, 0)
    X_all = X.clone()

    cam2ncam, s, _ = fit_unit_cube_median_quantile(X_all.numpy(), quantile_drop_threshold)
    pointmap_ncam = pointmap_3d.clone()
    pointmap_ncam[:, mask_tensor] = (
        pointmap_ncam[:, mask_tensor].permute(1, 0) @ cam2ncam[:3, :3].T + cam2ncam[:3, 3]
    ).permute(1, 0).float()
    pointmap_ncam = pointmap_ncam * mask_tensor.int()

    left, upper, right, lower = aug_bbox
    out_h, out_w = lower - upper, right - left
    H, W = pointmap_ncam.shape[1], pointmap_ncam.shape[2]
    pointmap_ncam_crop = torch.zeros((3, out_h, out_w), dtype=pointmap_ncam.dtype)

    src_top, src_bottom = max(upper, 0), min(lower, H)
    src_left, src_right = max(left, 0), min(right, W)
    dst_top, dst_left = src_top - upper, src_left - left

    pointmap_ncam_crop[:, dst_top:dst_top + (src_bottom - src_top),
                       dst_left:dst_left + (src_right - src_left)] = \
        pointmap_ncam[:, src_top:src_bottom, src_left:src_right]

    pointmap_ncam = F.interpolate(
        pointmap_ncam_crop[None], size=(image_size, image_size), mode='nearest'
    )[0].contiguous()

    pointmap = pointmap_ncam.clone()
    if clamp_range is not None:
        pointmap = pointmap.clamp(*clamp_range)

    return pointmap, image_resized, mask_resized, cam2ncam


# ---------------------------------------------------------------------------
# High-level single-view preprocessing
# ---------------------------------------------------------------------------

def preprocess_view(image, depth, mask, intrinsics,
                    quantile_drop_threshold=0.025,
                    clamp_range=(-2.0, 3.0),
                    mask_erosion_enabled=True, mask_erosion_params=None):
    """Preprocess a single view: erosion → pointmap → normalise → crop.

    Returns dict with 'image', 'pointmap', 'mask', 'cam2ncam'.
    """
    mask_eroded, _ = apply_mask_erosion(mask.copy(), enabled=mask_erosion_enabled,
                                        params=mask_erosion_params)
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
        pointmap_scene, pil_image, Image.fromarray(mask_eroded), valid,
        quantile_drop_threshold=quantile_drop_threshold,
        clamp_range=clamp_range,
    )

    return {
        'image': resized_image,
        'pointmap': pointmap_tensor,
        'mask': resized_mask,
        'cam2ncam': cam2ncam,
    }
