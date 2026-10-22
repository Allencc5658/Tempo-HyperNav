"""
UniGoal navigation engine wrapper.
Bridges the FastAPI service layer with the actual navigation code in src/.

Launches main.py as a subprocess with the correct config file per goal_type,
GPU isolation via CUDA_VISIBLE_DEVICES, and captures output for monitoring.
"""

import asyncio
import base64
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Optional

from app.config import settings
from app.models.schemas import GoalOption, TaskCreate

logger = logging.getLogger(__name__)


def _build_temp_config(
    base_config_path: Path,
    run_dir: Path,
    episode_id: int,
    gpu_id: int,
    snapshot_interval: int,
) -> Path:
    """
    Create a temporary YAML config that inherits from the base config
    but overrides output paths and GPU settings for this specific task.

    Why a temp config instead of CLI args?
    Because main.py does `args.update(config)` — YAML values OVERRIDE CLI args.
    So we must put our overrides directly in the YAML to guarantee they take effect.
    """
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override for single-episode, correct GPU, and output location
    cfg["num_eval_episodes"] = 1
    cfg["episode_id"] = episode_id
    # sim_gpu_id must be 0 because CUDA_VISIBLE_DEVICES already restricts to one GPU
    cfg["sim_gpu_id"] = 0
    cfg["visualize"] = 1
    cfg["dump_location"] = str(run_dir)
    cfg["experiment_id"] = "nav"
    cfg["snapshot_interval"] = snapshot_interval

    temp_path = run_dir / "config_override.yaml"
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return temp_path


async def run_navigation_task(
    task_id: str,
    goal: GoalOption,
    params: TaskCreate,
    gpu_id: int = 0,
) -> dict:
    """
    Execute a single UniGoal navigation episode via subprocess.

    Flow:
    1. Select base config file based on goal_type
       - ins-image → configs/config_habitat.yaml
       - text     → configs/config_textnav.yaml
       - objnav   → configs/config_objnav.yaml
    2. Generate a temp config that overrides:
       - num_eval_episodes=1, episode_id, sim_gpu_id=0, dump_location, visualize=1
    3. Launch `python main.py --config-file <temp> --goal_type <type> --episode_id <id>`
       with CUDA_VISIBLE_DEVICES=<gpu_id> in a subprocess
    4. Stream stdout/stderr to log file + WebSocket
    5. Parse results from checkpoint.json after process exits

    Returns:
        dict with keys: success, spl, total_steps, video_path, snapshot_dir
    """

    # ── Prepare directories ──
    run_dir = settings.RUNS_DIR / task_id
    graphs_dir = run_dir / "graphs"
    frames_dir = run_dir / "frames"
    logs_dir = run_dir / "logs"
    for d in [graphs_dir, frames_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Select config file based on goal_type ──
    goal_type_str = goal.goal_type.value if hasattr(goal.goal_type, "value") else str(goal.goal_type)
    base_config_rel = settings.GOAL_CONFIG_MAP.get(goal_type_str)
    if not base_config_rel:
        raise ValueError(f"No config mapping for goal_type='{goal_type_str}'. "
                         f"Available: {list(settings.GOAL_CONFIG_MAP.keys())}")
    base_config_path = settings.PROJECT_ROOT / base_config_rel
    if not base_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {base_config_path}")

    logger.info(f"[Task {task_id}] goal_type={goal_type_str} → config={base_config_rel}")

    # ── Generate temp config with overrides ──
    temp_config = _build_temp_config(
        base_config_path=base_config_path,
        run_dir=run_dir,
        episode_id=goal.episode_id,
        gpu_id=gpu_id,
        snapshot_interval=settings.SNAPSHOT_INTERVAL,
    )

    # ── Build subprocess command ──
    cmd = [
        sys.executable, "-u",
        str(settings.PROJECT_ROOT / "main.py"),
        "--config-file", str(temp_config),
        "--goal_type", goal_type_str,
        "--episode_id", str(goal.episode_id),
    ]
    # Web output dir — triggers frame + graph snapshot export in main.py
    cmd.extend(["--web_output_dir", str(run_dir)])
    # Ablation flags
    if params.disable_temporal:
        cmd.append("--disable_temporal")
    if params.disable_hypergraph:
        cmd.append("--disable_hypergraph")
    if params.disable_peek:
        cmd.append("--disable_peek")

    # ── GPU isolation ──
    worker_env = os.environ.copy()
    worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info(f"[Task {task_id}] CUDA_VISIBLE_DEVICES={gpu_id}")
    logger.info(f"[Task {task_id}] CMD: {' '.join(cmd)}")

    # ── Launch subprocess ──
    log_path = logs_dir / "navigation.log"
    log_file = open(log_path, "w", encoding="utf-8")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=worker_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(settings.PROJECT_ROOT),
    )
    logger.info(f"[Task {task_id}] Subprocess PID={proc.pid}")

    # ── Stream stdout to log file + optionally broadcast via WebSocket ──
    # The agent writes a visualization frame to outputs/tmp/v.jpg every step
    vis_frame_path = settings.PROJECT_ROOT / "outputs" / "tmp" / "v.jpg"

    # Throttle WebSocket broadcasts: push every WS_PUSH_EVERY_N steps.
    # Lower values = smoother UI but more network/CPU cost.
    # 3 is a good balance: ~0.6s wall-time gap at 5 Hz stepping.
    WS_PUSH_EVERY_N = 3
    _last_pushed_step = -WS_PUSH_EVERY_N  # ensure first step is pushed

    try:
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace")
            log_file.write(line)
            log_file.flush()
            # Parse step progress lines like "num timesteps 42, episode_id 0"
            if "num timesteps" in line:
                try:
                    from app.routers.ws import broadcast_progress
                    parts = line.strip().split(",")
                    step_str = parts[0].split()[-1]  # "42"
                    step_num = int(step_str.strip(","))

                    # Throttle: only broadcast every N steps
                    if step_num - _last_pushed_step < WS_PUSH_EVERY_N:
                        continue
                    _last_pushed_step = step_num

                    # Read latest visualization frame as base64
                    rgb_frame = None
                    try:
                        if vis_frame_path.exists():
                            with open(vis_frame_path, "rb") as vf:
                                rgb_frame = base64.b64encode(vf.read()).decode("ascii")
                    except Exception:
                        pass

                    # Read latest graph snapshot (for live scene graph in Monitor)
                    nodes_count, edges_count = 0, 0
                    current_confidence = None
                    graph_snapshot = None
                    try:
                        snap_files = sorted(graphs_dir.glob("step_*.json"))
                        if snap_files:
                            with open(snap_files[-1], "r") as sf:
                                snap = json.load(sf)
                            snap_nodes = snap.get("nodes", [])
                            snap_edges = snap.get("edges", [])
                            nodes_count = len(snap_nodes)
                            edges_count = len(snap_edges)
                            if snap_nodes:
                                current_confidence = max(
                                    n.get("confidence", 0) for n in snap_nodes
                                )
                            graph_snapshot = snap
                    except Exception:
                        pass

                    await broadcast_progress(task_id, {
                        "task_id": task_id,
                        "step": step_num,
                        "max_steps": 1000,
                        "status_text": f"navigating (step {step_num})",
                        "rgb_frame": rgb_frame,
                        "current_confidence": current_confidence,
                        "nodes_count": nodes_count,
                        "edges_count": edges_count,
                        "graph_snapshot": graph_snapshot,
                    })
                except Exception:
                    pass  # Best-effort, don't break navigation
    finally:
        log_file.close()

    return_code = await proc.wait()
    logger.info(f"[Task {task_id}] Process exited with code {return_code}")

    # ── Parse results ──
    result = {
        "success": 0.0,
        "spl": 0.0,
        "total_steps": 0,
        "video_path": None,
        "snapshot_dir": str(graphs_dir),
    }

    # main.py writes checkpoint.json and total.json in dump_location/experiment_id/log/
    result_log_dir = run_dir / "nav" / "log"
    checkpoint_path = result_log_dir / "checkpoint.json"
    total_path = result_log_dir / "total.json"

    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            results_list = ckpt.get("results", [])
            if results_list:
                last = results_list[-1]
                result["success"] = float(last.get("success", 0.0))
                result["spl"] = float(last.get("spl", 0.0))
            logger.info(f"[Task {task_id}] Parsed checkpoint: SR={result['success']}, SPL={result['spl']}")
        except Exception as e:
            logger.warning(f"[Task {task_id}] Failed to parse checkpoint: {e}")

    # Find the video file (main.py saves to visualization/videos/eps_XXXXXX.mp4)
    vis_dir = run_dir / "nav" / "visualization" / "videos"
    if vis_dir.exists():
        videos = list(vis_dir.glob("eps_*.mp4"))
        if videos:
            # Copy/link to standard location for static serving
            src_video = videos[0]
            dst_video = run_dir / "video.mp4"
            if not dst_video.exists():
                try:
                    os.link(str(src_video), str(dst_video))
                except OSError:
                    import shutil
                    shutil.copy2(str(src_video), str(dst_video))
            result["video_path"] = str(dst_video)
            logger.info(f"[Task {task_id}] Video: {dst_video}")

    if return_code != 0:
        logger.error(f"[Task {task_id}] Non-zero exit code: {return_code}. Check {log_path}")
        raise RuntimeError(f"Navigation subprocess failed (exit code {return_code}). See {log_path}")

    return result


def export_graph_snapshot(graph, step: int, output_dir: Path, agent_pose: Optional[dict] = None):
    """
    Serialize the current state of a Graph object to JSON.
    Called from within the navigation loop.

    IMPORTANT: This is a pure READ-ONLY function — it only reads graph state
    and writes to disk. It never modifies graph.nodes, graph.edges, or any
    internal data structures of the original Graph object.

    Field names match frontend GraphSnapshot type and schemas.GraphSnapshot.
    Source data comes from graph.export_for_hypergraph() + additional fields.
    """
    # --- Read nodes (matches ObjectNode fields) ---
    node_uid_map: dict = {}
    nodes_out = []
    for idx, node in enumerate(graph.nodes):
        uid = f"{node.caption}_{id(node)}"
        node_uid_map[node] = uid
        nodes_out.append({
            "uid": uid,
            "caption": node.caption or "",
            "center": list(node.center) if node.center is not None else [0, 0],
            "confidence": getattr(node, "confidence", 0.5),
            "n_obs": getattr(node, "n_obs", 0),
            "c_trend": getattr(node, "c_trend", 0.0),
            "is_in_fov": getattr(node, "is_in_FOV", False),
        })

    # --- Read edges (matches Edge fields) ---
    edges_out = []
    seen_edges: set = set()
    for node in graph.nodes:
        for edge in node.edges:
            edge_key = tuple(sorted([id(edge.node1), id(edge.node2)]))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                src_uid = node_uid_map.get(edge.node1)
                dst_uid = node_uid_map.get(edge.node2)
                if src_uid and dst_uid:
                    edges_out.append({
                        "src": src_uid,
                        "dst": dst_uid,
                        "weight": getattr(edge, "weight", 1.0),
                        "n_simul": getattr(edge, "n_simultaneous", 0),
                    })

    # --- Read hyperedges (from prob_map + hypergraph_reasoner if available) ---
    hyperedges_out = []

    # --- Read prob_map (posterior probabilities from hypergraph BP) ---
    prob_map_out = {}
    raw_prob = getattr(graph, "prob_map", {})
    for node_obj, prob in raw_prob.items():
        uid = node_uid_map.get(node_obj)
        if uid:
            prob_map_out[uid] = float(prob)

    # --- Assemble snapshot ---
    snapshot = {
        "step": step,
        "agent_pose": agent_pose or {"x": 0, "y": 0, "heading": 0},
        "nodes": nodes_out,
        "edges": edges_out,
        "hyperedges": hyperedges_out,
        "prob_map": prob_map_out,
        "frontiers": [],
        "delta_G": None,
    }

    output_path = output_dir / f"step_{step:05d}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    return output_path
