import json
import math
from pathlib import Path


def _as_point(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (x, y)


def _point_distance(left, right):
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _bucket_point(point, bucket_cells):
    if point is None:
        return None
    bucket = max(float(bucket_cells), 1.0)
    return (int(round(point[0] / bucket)), int(round(point[1] / bucket)))


def analyze_frontier_snapshots(snapshots, revisit_radius_cells=20, oscillation_bucket_cells=20):
    """Summarize frontier-selection behavior from exported graph snapshots."""
    selected_points = []
    belief_values = []
    positive_values = []
    negative_values = []
    distance_cost_values = []
    selected_term_counts = {}
    selection_modes = {}
    negative_evidence_decisions = 0
    belief_decisions = 0
    rescue_decisions = 0

    for snapshot in snapshots:
        frontier_debug = snapshot.get("frontier_debug") or {}
        selected = _as_point(frontier_debug.get("selected"))
        if selected is None:
            selected = _as_point(snapshot.get("selected_global_goal"))
        if selected is None:
            selected = _as_point(snapshot.get("semantic_direct_goal"))
        if selected is None:
            continue
        selected_points.append(selected)

        mode = frontier_debug.get("selection_mode")
        if mode:
            selection_modes[mode] = selection_modes.get(mode, 0) + 1
        if frontier_debug.get("rescue_mode"):
            rescue_decisions += 1

        terms = frontier_debug.get("selected_terms") or {}
        for key, value in terms.items():
            if value:
                selected_term_counts[key] = selected_term_counts.get(key, 0) + 1
        if any(float(terms.get(key, 0.0) or 0.0) < 0.0 for key in ("arrival_reject", "peek", "exhausted")):
            negative_evidence_decisions += 1

        belief = frontier_debug.get("belief") or {}
        if belief:
            belief_decisions += 1
            for key, target in (
                ("selected_belief", belief_values),
                ("selected_positive", positive_values),
                ("selected_negative", negative_values),
                ("selected_distance_cost", distance_cost_values),
            ):
                value = belief.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    target.append(float(value))

    switch_count = 0
    for prev, cur in zip(selected_points, selected_points[1:]):
        if _point_distance(prev, cur) > float(revisit_radius_cells):
            switch_count += 1

    revisit_count = 0
    seen = []
    for point in selected_points:
        if any(_point_distance(point, old) <= float(revisit_radius_cells) for old in seen):
            revisit_count += 1
        seen.append(point)

    buckets = [_bucket_point(point, oscillation_bucket_cells) for point in selected_points]
    oscillation_count = 0
    for idx in range(3, len(buckets)):
        a, b, c, d = buckets[idx - 3 : idx + 1]
        if a is not None and b is not None and a == c and b == d and a != b:
            oscillation_count += 1

    max_same_goal_streak = 0
    current_streak = 0
    last_bucket = None
    for bucket in buckets:
        if bucket is None:
            continue
        if bucket == last_bucket:
            current_streak += 1
        else:
            current_streak = 1
            last_bucket = bucket
        max_same_goal_streak = max(max_same_goal_streak, current_streak)

    decisions = len(selected_points)
    return {
        "num_snapshots": len(snapshots),
        "num_frontier_decisions": decisions,
        "frontier_switch_count": switch_count,
        "frontier_switch_rate": switch_count / max(decisions - 1, 1),
        "frontier_revisit_count": revisit_count,
        "frontier_revisit_rate": revisit_count / max(decisions, 1),
        "frontier_oscillation_count": oscillation_count,
        "max_same_goal_streak": max_same_goal_streak,
        "negative_evidence_decisions": negative_evidence_decisions,
        "belief_decisions": belief_decisions,
        "rescue_decisions": rescue_decisions,
        "selected_term_counts": dict(sorted(selected_term_counts.items())),
        "selection_modes": dict(sorted(selection_modes.items())),
        "mean_selected_belief": _mean_or_none(belief_values),
        "mean_selected_positive": _mean_or_none(positive_values),
        "mean_selected_negative": _mean_or_none(negative_values),
        "mean_selected_distance_cost": _mean_or_none(distance_cost_values),
    }


def summarize_checkpoint(checkpoint):
    results = checkpoint.get("results") if isinstance(checkpoint, dict) else None
    if not isinstance(results, list):
        results = []
    successes = [
        float(item.get("success", 0.0) or 0.0)
        for item in results
        if isinstance(item, dict)
    ]
    spls = [
        float(item.get("spl", 0.0) or 0.0)
        for item in results
        if isinstance(item, dict)
    ]
    distances = []
    num_steps = []
    path_lengths = []
    failed_close_count = 0
    failed_with_distance_count = 0
    timeout_like_count = 0
    stopped_count = 0
    failed_stopped_count = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        success = float(item.get("success", 0.0) or 0.0)
        if item.get("stopped") is True:
            stopped_count += 1
            if success <= 0.0:
                failed_stopped_count += 1
        steps = item.get("num_steps")
        if isinstance(steps, (int, float)) and math.isfinite(float(steps)):
            steps = int(steps)
            num_steps.append(steps)
            max_steps = item.get("max_episode_steps")
            if isinstance(max_steps, (int, float)) and float(max_steps) > 0:
                if steps >= int(float(max_steps) * 0.9):
                    timeout_like_count += 1
        path_length = item.get("path_length")
        if isinstance(path_length, (int, float)) and math.isfinite(float(path_length)):
            path_lengths.append(float(path_length))
        distance = item.get("distance_to_goal")
        if isinstance(distance, (int, float)) and math.isfinite(float(distance)):
            distance = float(distance)
            distances.append(distance)
            if success <= 0.0:
                failed_with_distance_count += 1
                success_dist = item.get("success_dist", 1.0)
                try:
                    success_dist = float(success_dist)
                except (TypeError, ValueError):
                    success_dist = 1.0
                if distance <= success_dist:
                    failed_close_count += 1
    return {
        "num_completed_episodes": len(results),
        "success_rate": _mean_or_none(successes),
        "spl": _mean_or_none(spls),
        "failed_episode_count": sum(1 for value in successes if value <= 0.0),
        "mean_distance_to_goal": _mean_or_none(distances),
        "mean_num_steps": _mean_or_none(num_steps),
        "mean_path_length": _mean_or_none(path_lengths),
        "timeout_like_count": timeout_like_count,
        "stopped_count": stopped_count,
        "failed_stopped_count": failed_stopped_count,
        "failed_with_distance_count": failed_with_distance_count,
        "failed_close_count": failed_close_count,
    }


def summarize_goat_trace(records):
    valid = [item for item in records if isinstance(item, dict)]
    if not valid:
        return {
            "num_goat_subtasks": 0,
            "goat_subtask_sr": None,
            "goat_subtask_spl": None,
            "goat_by_type": {},
        }
    successes = [float(item.get("success", 0.0) or 0.0) for item in valid]
    spls = [float(item.get("spl", 0.0) or 0.0) for item in valid]
    by_type = {}
    for item in valid:
        goal_type = str(item.get("goal_type", "unknown"))
        bucket = by_type.setdefault(goal_type, {"success": [], "spl": []})
        bucket["success"].append(float(item.get("success", 0.0) or 0.0))
        bucket["spl"].append(float(item.get("spl", 0.0) or 0.0))
    by_type_summary = {
        goal_type: {
            "count": len(values["success"]),
            "sr": _mean_or_none(values["success"]),
            "spl": _mean_or_none(values["spl"]),
        }
        for goal_type, values in sorted(by_type.items())
    }
    return {
        "num_goat_subtasks": len(valid),
        "goat_subtask_sr": _mean_or_none(successes),
        "goat_subtask_spl": _mean_or_none(spls),
        "goat_by_type": by_type_summary,
    }


def load_run_diagnostics(run_dir, revisit_radius_cells=20, oscillation_bucket_cells=20):
    run_path = Path(run_dir)
    snapshots = _load_snapshots(run_path / "graphs")
    log_path = _resolve_log_dir(run_path)
    frontier_trace = _load_jsonl(log_path / "frontier_trace.jsonl")
    frontier_records = frontier_trace if frontier_trace else snapshots
    frontier_summary = analyze_frontier_snapshots(
        frontier_records,
        revisit_radius_cells=revisit_radius_cells,
        oscillation_bucket_cells=oscillation_bucket_cells,
    )
    frontier_summary["source"] = "frontier_trace" if frontier_trace else "snapshots"
    frontier_summary["num_frontier_trace_records"] = len(frontier_trace)
    checkpoint = _load_json(log_path / "checkpoint.json")
    goat_trace = _load_jsonl(log_path / "goat_trace.jsonl")
    return {
        "run_dir": str(run_path),
        "frontier": frontier_summary,
        "checkpoint": summarize_checkpoint(checkpoint or {}),
        "goat": summarize_goat_trace(goat_trace),
    }


def _resolve_log_dir(run_path):
    for candidate in (
        run_path / "nav" / "log",
        run_path / "log",
        run_path,
    ):
        if candidate.is_dir():
            return candidate
    return run_path / "nav" / "log"


def _mean_or_none(values):
    vals = [float(value) for value in values if math.isfinite(float(value))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _load_json(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_jsonl(path):
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _load_snapshots(graph_dir):
    if not graph_dir.is_dir():
        return []
    snapshots = []
    for path in sorted(graph_dir.glob("*.json")):
        snapshot = _load_json(path)
        if isinstance(snapshot, dict):
            snapshots.append(snapshot)
    return snapshots
