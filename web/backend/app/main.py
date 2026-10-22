"""
UniGoal Visualization System - Backend Entry Point
FastAPI application serving navigation tasks, results, and real-time monitoring.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.database import init_db
from app.routers import tasks, results, ws
from app.services.task_queue import start_workers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    await init_db()
    # Ensure output directories exist
    os.makedirs(settings.RUNS_DIR, exist_ok=True)
    os.makedirs(settings.GOALS_DIR, exist_ok=True)
    # Start background navigation workers
    await start_workers()
    yield
    # Cleanup (if needed)


app = FastAPI(
    title="UniGoal Visualization System",
    description="Interactive visualization & monitoring for UniGoal navigation agent.",
    version="0.1.0",
    lifespan=lifespan,
)

# --- CORS (allow frontend dev server) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static file serving ---
# Serve run artifacts (videos, snapshots, frames)
os.makedirs(settings.RUNS_DIR, exist_ok=True)
app.mount("/static/runs", StaticFiles(directory=settings.RUNS_DIR), name="runs")

# Serve goal images
os.makedirs(settings.GOALS_DIR, exist_ok=True)
app.mount("/static/goals", StaticFiles(directory=settings.GOALS_DIR), name="goals")

# --- Routers ---
app.include_router(tasks.router, prefix="/api/tasks", tags=["Tasks"])
app.include_router(results.router, prefix="/api/results", tags=["Results"])
app.include_router(ws.router, prefix="/ws", tags=["WebSocket"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": app.version}
