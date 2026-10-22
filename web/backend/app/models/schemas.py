"""
Pydantic models for API request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────

class GoalType(str, Enum):
    INS_IMAGE = "ins-image"
    TEXT = "text"
    OBJNAV = "objnav"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# ── Goal Catalog ───────────────────────────────────────

class GoalOption(BaseModel):
    """A pre-defined goal available for selection."""
    id: str
    display_name: str
    goal_type: GoalType
    goal_image: Optional[str] = None       # relative path to image
    goal_text: Optional[str] = None
    episode_id: int
    scene_id: str
    preview_image: Optional[str] = None    # scene preview thumbnail
    difficulty: str = "medium"
    description: Optional[str] = None      # human-readable scene description


# ── Task ───────────────────────────────────────────────

class TaskCreate(BaseModel):
    """Request body for creating a new navigation task."""
    goal_id: str = Field(..., description="ID from goal_catalog.json")
    # Ablation flags
    disable_temporal: bool = False
    disable_hypergraph: bool = False
    disable_peek: bool = False


class TaskOut(BaseModel):
    """Task record returned to the frontend."""
    id: str
    goal_id: str
    goal_type: GoalType
    display_name: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    # Result metrics (filled when done)
    success: Optional[float] = None
    spl: Optional[float] = None
    total_steps: Optional[int] = None
    # Paths to artifacts (relative, served via /static/runs/)
    video_path: Optional[str] = None
    snapshot_dir: Optional[str] = None

    class Config:
        from_attributes = True


class TaskProgress(BaseModel):
    """WebSocket message pushed during navigation."""
    task_id: str
    step: int
    max_steps: int
    # Base64 encoded current RGB frame (optional, can be large)
    rgb_frame: Optional[str] = None
    # Lightweight text status
    status_text: str = ""
    # Live metrics
    current_confidence: Optional[float] = None
    nodes_count: int = 0
    edges_count: int = 0


# ── Results ────────────────────────────────────────────

class GraphSnapshot(BaseModel):
    """Schema for a single graph snapshot JSON file."""
    step: int
    agent_pose: dict                                 # {x, y, heading}
    nodes: List[dict]                                # [{uid, caption, center, confidence, ...}]
    edges: List[dict]                                # [{src, dst, weight, ...}]
    hyperedges: List[dict] = []                      # [{name, members, confidence}]
    prob_map: Dict[str, float] = {}                  # {node_uid: posterior}
    frontiers: List[dict] = []                       # [{pos, score, temporal_bonus}]
    delta_G: Optional[dict] = None                   # {new_nodes, disappeared, confirmed, declining}


class EpisodeMetrics(BaseModel):
    """Aggregated metrics for a completed episode."""
    task_id: str
    success: float
    spl: float
    total_steps: int
    num_nodes_final: int = 0
    num_edges_final: int = 0
    avg_confidence: float = 0.0
    snapshot_count: int = 0


class CompareRequest(BaseModel):
    """Request body for comparing multiple tasks."""
    task_ids: List[str]
