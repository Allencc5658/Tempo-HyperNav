"""
Central configuration for the backend.
All paths are relative to the UniGoal project root.
"""

import os
from pathlib import Path
from typing import Dict, List
from pydantic_settings import BaseSettings


# UniGoal project root (two levels up from this file: web/backend/app/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # --- Paths ---
    PROJECT_ROOT: Path = _PROJECT_ROOT
    UNIGOAL_SRC: Path = _PROJECT_ROOT / "src"
    UNIGOAL_CONFIG: Path = _PROJECT_ROOT / "configs" / "config_habitat.yaml"
    DATA_DIR: Path = _PROJECT_ROOT / "web" / "storage"
    RUNS_DIR: Path = _PROJECT_ROOT / "web" / "storage" / "runs"
    GOALS_DIR: Path = _PROJECT_ROOT / "web" / "storage" / "goals"
    GOAL_CATALOG: Path = _PROJECT_ROOT / "web" / "storage" / "goal_catalog.json"
    DB_PATH: Path = _PROJECT_ROOT / "web" / "storage" / "unigoal_vis.db"

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- Navigation engine ---
    SNAPSHOT_INTERVAL: int = 10          # Export graph snapshot every N steps
    MAX_CONCURRENT_TASKS: int = 1        # Max parallel navigation tasks
    # Physical GPU IDs to use. Each worker gets one GPU in round-robin.
    # Override: UNIGOAL_VIS_DEFAULT_GPU_IDS='[2,3]'  or  ='[3]' for single GPU
    # These are PHYSICAL GPU IDs (same as nvidia-smi), NOT CUDA device indices.
    DEFAULT_GPU_IDS: List[int] = [0]

    # Config file per goal_type (relative to PROJECT_ROOT)
    # ins-image uses the default config; text/objnav use their own task configs.
    GOAL_CONFIG_MAP: Dict[str, str] = {
        "ins-image": "configs/config_habitat.yaml",
        "text":      "configs/config_textnav.yaml",
        "objnav":    "configs/config_objnav.yaml",
    }

    # --- WebSocket ---
    WS_FRAME_INTERVAL: int = 3           # Push frame to client every N steps

    class Config:
        env_prefix = "UNIGOAL_VIS_"


settings = Settings()
