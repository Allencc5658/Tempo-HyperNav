"""
Task management API: create, list, get, cancel navigation tasks.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.db.database import get_db
from app.models.schemas import GoalOption, GoalType, TaskCreate, TaskOut, TaskStatus
from app.services.task_queue import submit_task

router = APIRouter()


def _load_goal_catalog() -> Dict[str, GoalOption]:
    """Load pre-defined goals from catalog file."""
    catalog_path = settings.GOAL_CATALOG
    if not catalog_path.exists():
        return {}
    with open(catalog_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["id"]: GoalOption(**item) for item in items}


@router.get("/catalog", response_model=List[GoalOption])
async def list_goals():
    """Return all pre-defined goals available for selection."""
    catalog = _load_goal_catalog()
    return list(catalog.values())


@router.post("/", response_model=TaskOut, status_code=201)
async def create_task(body: TaskCreate):
    """Submit a new navigation task."""
    catalog = _load_goal_catalog()
    goal = catalog.get(body.goal_id)
    if not goal:
        raise HTTPException(404, f"Goal '{body.goal_id}' not found in catalog")

    task_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO tasks
               (id, goal_id, goal_type, display_name, status, created_at,
                disable_temporal, disable_hypergraph, disable_peek)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (task_id, body.goal_id, goal.goal_type.value, goal.display_name,
             TaskStatus.QUEUED.value, now,
             int(body.disable_temporal), int(body.disable_hypergraph), int(body.disable_peek)),
        )
        await db.commit()
    finally:
        await db.close()

    # Enqueue for async execution
    await submit_task(task_id, goal, body)

    return TaskOut(
        id=task_id,
        goal_id=body.goal_id,
        goal_type=goal.goal_type,
        display_name=goal.display_name,
        status=TaskStatus.QUEUED,
        created_at=now,
    )


@router.get("/", response_model=List[TaskOut])
async def list_tasks(status: Optional[TaskStatus] = None, limit: int = 50):
    """List tasks, optionally filtered by status."""
    db = await get_db()
    try:
        if status:
            cursor = await db.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status.value, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    return [_row_to_task(r) for r in rows]


@router.get("/{task_id}", response_model=TaskOut)
async def get_task(task_id: str):
    """Get a single task by ID."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
    finally:
        await db.close()

    if not row:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return _row_to_task(row)


@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a queued or running task."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT status FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, f"Task '{task_id}' not found")
        if row["status"] not in (TaskStatus.QUEUED.value, TaskStatus.RUNNING.value):
            raise HTTPException(400, f"Cannot cancel task with status '{row['status']}'")
        await db.execute(
            "UPDATE tasks SET status = ? WHERE id = ?",
            (TaskStatus.FAILED.value, task_id),
        )
        await db.commit()
    finally:
        await db.close()

    return {"message": f"Task '{task_id}' cancelled"}


@router.delete("/{task_id}/permanent")
async def delete_task(task_id: str):
    """Permanently delete a finished (done/failed) task and its data."""
    import shutil

    db = await get_db()
    try:
        cursor = await db.execute("SELECT status FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(404, f"Task '{task_id}' not found")
        if row["status"] in (TaskStatus.QUEUED.value, TaskStatus.RUNNING.value):
            raise HTTPException(400, "Cannot delete a task that is still queued or running. Cancel it first.")
        await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        await db.commit()
    finally:
        await db.close()

    # Remove filesystem artifacts
    run_dir = settings.RUNS_DIR / task_id
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    return {"message": f"Task '{task_id}' deleted"}


# ── Helpers ────────────────────────────────────────────

def _row_to_task(row) -> TaskOut:
    return TaskOut(
        id=row["id"],
        goal_id=row["goal_id"],
        goal_type=GoalType(row["goal_type"]),
        display_name=row["display_name"],
        status=TaskStatus(row["status"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        success=row["success"],
        spl=row["spl"],
        total_steps=row["total_steps"],
        video_path=row["video_path"],
        snapshot_dir=row["snapshot_dir"],
    )
