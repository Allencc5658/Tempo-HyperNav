"""
Results API: query completed task artifacts — videos, graph snapshots, metrics.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.db.database import get_db
from app.models.schemas import CompareRequest, EpisodeMetrics, GraphSnapshot, TaskStatus

router = APIRouter()


@router.get("/{task_id}/metrics", response_model=EpisodeMetrics)
async def get_metrics(task_id: str):
    """Return aggregated metrics for a completed task."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
    finally:
        await db.close()

    if not row:
        raise HTTPException(404, "Task not found")
    if row["status"] != TaskStatus.DONE.value:
        raise HTTPException(400, "Task not yet completed")

    # Count snapshots on disk
    snap_dir = settings.RUNS_DIR / task_id / "graphs"
    snapshot_count = len(list(snap_dir.glob("*.json"))) if snap_dir.exists() else 0

    return EpisodeMetrics(
        task_id=task_id,
        success=row["success"] or 0.0,
        spl=row["spl"] or 0.0,
        total_steps=row["total_steps"] or 0,
        snapshot_count=snapshot_count,
    )


@router.get("/{task_id}/snapshots")
async def list_snapshots(task_id: str):
    """List all available graph snapshot steps for a task."""
    snap_dir = settings.RUNS_DIR / task_id / "graphs"
    if not snap_dir.exists():
        return []
    files = sorted(snap_dir.glob("step_*.json"))
    return [{"step": int(f.stem.split("_")[1]), "filename": f.name} for f in files]


@router.get("/{task_id}/snapshots/{step}", response_model=GraphSnapshot)
async def get_snapshot(task_id: str, step: int):
    """Return a single graph snapshot by step number."""
    snap_path = settings.RUNS_DIR / task_id / "graphs" / f"step_{step:05d}.json"
    if not snap_path.exists():
        raise HTTPException(404, f"Snapshot at step {step} not found")
    with open(snap_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return GraphSnapshot(**data)


@router.get("/{task_id}/video")
async def get_video(task_id: str):
    """Stream the episode video file."""
    video_path = settings.RUNS_DIR / task_id / "video.mp4"
    if not video_path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(video_path, media_type="video/mp4")


@router.get("/{task_id}/frames")
async def list_frames(task_id: str):
    """List saved key frames (if any)."""
    frames_dir = settings.RUNS_DIR / task_id / "frames"
    if not frames_dir.exists():
        return []
    files = sorted(frames_dir.glob("*.jpg"))
    return [{"step": int(f.stem.split("_")[1]), "url": f"/static/runs/{task_id}/frames/{f.name}"} for f in files]


@router.post("/compare")
async def compare_tasks(body: CompareRequest):
    """Return metrics for multiple tasks side-by-side (for ablation comparison)."""
    task_ids = body.task_ids
    db = await get_db()
    try:
        placeholders = ",".join("?" for _ in task_ids)
        cursor = await db.execute(
            f"SELECT * FROM tasks WHERE id IN ({placeholders}) AND status = 'done'",
            task_ids,
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    return [
        {
            "task_id": r["id"],
            "display_name": r["display_name"],
            "success": r["success"],
            "spl": r["spl"],
            "total_steps": r["total_steps"],
            "disable_temporal": bool(r["disable_temporal"]),
            "disable_hypergraph": bool(r["disable_hypergraph"]),
            "disable_peek": bool(r["disable_peek"]),
        }
        for r in rows
    ]
