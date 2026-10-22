"""
Async task queue for navigation job scheduling.
Uses asyncio.Queue for simplicity (single server, limited concurrency).
For production / multi-server, replace with Celery + Redis.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from app.config import settings
from app.models.schemas import GoalOption, TaskCreate, TaskStatus

logger = logging.getLogger(__name__)

# In-memory async queue
_task_queue: asyncio.Queue = asyncio.Queue()

# Track running workers
_running_tasks: Dict[str, asyncio.Task] = {}


async def submit_task(task_id: str, goal: GoalOption, params: TaskCreate):
    """Enqueue a navigation task for async execution."""
    await _task_queue.put({
        "task_id": task_id,
        "goal": goal,
        "params": params,
    })
    logger.info(f"Task {task_id} enqueued (queue size: {_task_queue.qsize()})")


async def start_workers(num_workers: Optional[int] = None):
    """
    Start N worker coroutines that consume from the queue.
    Call this from app lifespan startup.
    """
    n = num_workers or settings.MAX_CONCURRENT_TASKS
    gpu_ids = settings.DEFAULT_GPU_IDS
    for i in range(n):
        gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids else 0
        asyncio.create_task(_worker(worker_id=i, gpu_id=gpu_id))
        logger.info(f"Spawned worker {i} → GPU {gpu_id}")


async def _worker(worker_id: int, gpu_id: int):
    """Single worker loop: take a task from queue and run it."""
    from app.services.engine import run_navigation_task

    logger.info(f"Worker {worker_id} started on GPU {gpu_id}")
    while True:
        item = await _task_queue.get()
        task_id = item["task_id"]
        logger.info(f"Worker {worker_id} picked up task {task_id}")

        try:
            await _update_task_status(task_id, TaskStatus.RUNNING)
            result = await run_navigation_task(
                task_id=task_id,
                goal=item["goal"],
                params=item["params"],
                gpu_id=gpu_id,
            )
            await _update_task_result(task_id, result)
        except Exception as e:
            logger.exception(f"Task {task_id} failed: {e}")
            await _update_task_status(task_id, TaskStatus.FAILED)
        finally:
            _task_queue.task_done()


async def _update_task_status(task_id: str, status: TaskStatus):
    """Update task status in the database."""
    from app.db.database import get_db

    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        if status == TaskStatus.RUNNING:
            await db.execute(
                "UPDATE tasks SET status = ?, started_at = ? WHERE id = ?",
                (status.value, now, task_id),
            )
        else:
            await db.execute(
                "UPDATE tasks SET status = ?, finished_at = ? WHERE id = ?",
                (status.value, now, task_id),
            )
        await db.commit()
    finally:
        await db.close()


async def _update_task_result(task_id: str, result: dict):
    """Write final metrics + artifact paths back to the database."""
    from app.db.database import get_db

    db = await get_db()
    try:
        await db.execute(
            """UPDATE tasks SET
                status = ?, finished_at = ?,
                success = ?, spl = ?, total_steps = ?,
                video_path = ?, snapshot_dir = ?
               WHERE id = ?""",
            (
                TaskStatus.DONE.value,
                datetime.now(timezone.utc).isoformat(),
                result.get("success", 0.0),
                result.get("spl", 0.0),
                result.get("total_steps", 0),
                result.get("video_path"),
                result.get("snapshot_dir"),
                task_id,
            ),
        )
        await db.commit()
    finally:
        await db.close()
