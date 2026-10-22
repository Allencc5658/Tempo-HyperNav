import numpy as np


def _coerce_vector(values, length, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.full(length, float(arr), dtype=float)
    if arr.ndim != 1 or len(arr) != length:
        raise ValueError(f"{name} must be a 1-D vector with length {length}")
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _normalize_nonnegative(values):
    arr = np.asarray(values, dtype=float)
    arr = np.maximum(arr, 0.0)
    max_value = float(np.max(arr)) if arr.size else 0.0
    if max_value <= 1e-8:
        return np.zeros_like(arr, dtype=float)
    return arr / max_value


def _noisy_or(components, length):
    if not components:
        return np.zeros(length, dtype=float)
    keep_missing = np.ones(length, dtype=float)
    for comp in components:
        keep_missing *= 1.0 - np.clip(comp, 0.0, 1.0)
    return 1.0 - keep_missing


def compute_frontier_belief_utility(
    distances_m,
    positive_evidence=None,
    negative_evidence=None,
    prior=0.35,
    positive_logit_weight=1.6,
    negative_logit_weight=1.8,
    distance_cost_weight=0.25,
    utility_weight=1.5,
):
    """Fuse frontier evidence into a bounded log-odds belief utility.

    Existing navigation code often mixes semantic, temporal, and verifier
    feedback as raw additive scores. This helper gives those heterogeneous
    signals a common interpretation:

    - positive evidence increases the target-existence log-odds near a frontier;
    - negative evidence decreases it;
    - distance is treated as a cost only when evidence exists.

    The returned utility is centered at the prior, so no evidence produces a
    zero adjustment instead of another hidden distance heuristic.
    """
    distances = np.asarray(distances_m, dtype=float)
    if distances.ndim != 1:
        distances = distances.reshape(-1)
    n = len(distances)
    if n == 0:
        empty = np.zeros(0, dtype=float)
        return {
            "utility": empty,
            "belief": empty,
            "positive": empty,
            "negative": empty,
            "distance_cost": empty,
        }

    prior = float(np.clip(prior, 1e-4, 1.0 - 1e-4))
    pos_components = []
    for name, values in (positive_evidence or {}).items():
        vec = _coerce_vector(values, n, f"positive_evidence[{name}]")
        pos_components.append(_normalize_nonnegative(vec))
    neg_components = []
    for name, values in (negative_evidence or {}).items():
        vec = _coerce_vector(values, n, f"negative_evidence[{name}]")
        neg_components.append(_normalize_nonnegative(np.abs(np.minimum(vec, 0.0))))

    positive = _noisy_or(pos_components, n)
    negative = _noisy_or(neg_components, n)

    finite = np.isfinite(distances)
    distance_cost = np.zeros(n, dtype=float)
    if np.any(finite):
        finite_dist = distances[finite]
        d_min = float(np.min(finite_dist))
        d_max = float(np.max(finite_dist))
        if d_max > d_min + 1e-8:
            distance_cost[finite] = (finite_dist - d_min) / (d_max - d_min)
        distance_cost[~finite] = 1.0
    else:
        distance_cost[:] = 1.0

    base_logit = np.log(prior / (1.0 - prior))
    logits = (
        base_logit
        + float(positive_logit_weight) * positive
        - float(negative_logit_weight) * negative
    )
    belief = 1.0 / (1.0 + np.exp(-logits))
    evidence_present = np.maximum(positive, negative)
    utility = (
        float(utility_weight) * (belief - prior)
        - float(distance_cost_weight) * distance_cost * evidence_present
    )

    return {
        "utility": utility.astype(float),
        "belief": belief.astype(float),
        "positive": positive.astype(float),
        "negative": negative.astype(float),
        "distance_cost": distance_cost.astype(float),
    }


def compute_arrival_reject_penalty(
    frontier_locations,
    rejected_regions,
    map_resolution_cm,
    penalty_weight=3.0,
    radius_m=2.0,
    enabled=True,
):
    """Compute a linear negative-evidence penalty for rejected goal regions."""
    frontiers = np.asarray(frontier_locations, dtype=float)
    if frontiers.ndim == 1:
        frontiers = frontiers.reshape(-1, 2)
    num_frontiers = len(frontiers)
    penalty_scores = np.zeros(num_frontiers, dtype=float)
    if not enabled or num_frontiers == 0:
        return penalty_scores

    rejected = np.asarray([] if rejected_regions is None else rejected_regions, dtype=float)
    if rejected.size == 0:
        return penalty_scores
    if rejected.ndim == 1:
        rejected = rejected.reshape(-1, 2)

    radius_cells = float(radius_m) * 100.0 / float(map_resolution_cm)
    if radius_cells <= 0 or penalty_weight <= 0:
        return penalty_scores

    for region in rejected:
        deltas = frontiers[:, :2] - region[:2]
        dists = np.linalg.norm(deltas, axis=1)
        inside = dists < radius_cells
        penalty_scores[inside] -= float(penalty_weight) * (
            1.0 - dists[inside] / radius_cells
        )

    return penalty_scores
