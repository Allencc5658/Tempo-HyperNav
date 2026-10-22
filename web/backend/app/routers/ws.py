"""
WebSocket endpoint for real-time navigation monitoring.
Clients subscribe to a task_id and receive step-by-step progress.
"""

import asyncio
import json
import logging
from collections import defaultdict
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)

# Active WebSocket connections grouped by task_id
_connections: Dict[str, List[WebSocket]] = defaultdict(list)

# Ping interval in seconds — keeps SSH tunnels / proxies alive
_PING_INTERVAL = 15


@router.websocket("/monitor/{task_id}")
async def monitor_task(websocket: WebSocket, task_id: str):
    """
    Client connects to receive real-time updates for a specific task.
    Messages are pushed by the navigation worker via `broadcast_progress()`.
    The handler also sends periodic pings to keep the connection alive
    through SSH tunnels and reverse proxies.
    """
    await websocket.accept()
    _connections[task_id].append(websocket)
    logger.info(f"[WS] Client connected for task {task_id} "
                f"(total: {len(_connections[task_id])})")
    try:
        while True:
            # Wait for client message OR ping timeout — whichever comes first
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=_PING_INTERVAL
                )
                # Client sent a message — echo ack
                await websocket.send_text(json.dumps({"ack": data}))
            except asyncio.TimeoutError:
                # No client message within timeout — send keepalive ping
                await websocket.send_text(json.dumps({"type": "ping"}))
    except (WebSocketDisconnect, Exception) as exc:
        # Catch ALL exceptions so the server never crashes from a WS handler.
        # WebSocketDisconnect is normal; others are logged as warnings.
        if not isinstance(exc, WebSocketDisconnect):
            logger.warning(f"[WS] Unexpected error for task {task_id}: {exc}")
    finally:
        # Safe removal — the socket may have been removed by broadcast already
        try:
            _connections[task_id].remove(websocket)
        except ValueError:
            pass
        if not _connections.get(task_id):
            _connections.pop(task_id, None)
        logger.info(f"[WS] Client disconnected for task {task_id} "
                    f"(remaining: {len(_connections.get(task_id, []))})")


async def broadcast_progress(task_id: str, message: dict):
    """
    Called by the navigation worker to push progress to all connected clients.
    This is the bridge between the engine and the frontend.
    """
    clients = _connections.get(task_id, [])
    if not clients:
        return
    payload = json.dumps(message)
    disconnected = []
    for ws in clients:
        try:
            await ws.send_text(payload)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        try:
            clients.remove(ws)
        except ValueError:
            pass
