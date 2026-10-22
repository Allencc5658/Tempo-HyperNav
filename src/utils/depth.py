import numpy as np


def habitat_depth_to_metric(depth, min_depth, max_depth, far_clip_value=None):
    """Convert Habitat depth observations to metric meters."""
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
        depth_arr = depth_arr[..., 0]
    else:
        depth_arr = depth_arr.copy()

    finite = np.isfinite(depth_arr)
    if np.any(finite):
        finite_max = float(np.nanmax(depth_arr[finite]))
        looks_normalized = finite_max <= 1.5
    else:
        looks_normalized = True

    if looks_normalized:
        norm_depth = np.clip(depth_arr, 0.0, 1.0)
        metric = float(min_depth) + norm_depth * (float(max_depth) - float(min_depth))
        far_mask = norm_depth >= 0.99
    else:
        metric = np.clip(depth_arr, float(min_depth), float(max_depth))
        far_mask = depth_arr >= float(max_depth) * 0.99

    metric[~finite] = float(max_depth)
    if far_clip_value is not None:
        metric[far_mask] = float(far_clip_value)
    return metric.astype(np.float32, copy=False)


def habitat_depth_to_cm(depth, min_depth, max_depth, far_clip_value_m=None):
    return habitat_depth_to_metric(
        depth,
        min_depth,
        max_depth,
        far_clip_value=far_clip_value_m,
    ) * 100.0
