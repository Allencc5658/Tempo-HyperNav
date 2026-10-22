"""
SQLite database via aiosqlite for async access.
Stores task metadata only; heavy artifacts (videos, snapshots) live on filesystem.
"""

import aiosqlite
from app.config import settings

_DB_PATH = str(settings.DB_PATH)


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(_DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


async def init_db():
    """Create tables if they don't exist."""
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id              TEXT PRIMARY KEY,
                goal_id         TEXT NOT NULL,
                goal_type       TEXT NOT NULL,
                display_name    TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'queued',
                created_at      TEXT NOT NULL,
                started_at      TEXT,
                finished_at     TEXT,
                success         REAL,
                spl             REAL,
                total_steps     INTEGER,
                video_path      TEXT,
                snapshot_dir    TEXT,
                -- Ablation flags
                disable_temporal   INTEGER DEFAULT 0,
                disable_hypergraph INTEGER DEFAULT 0,
                disable_peek       INTEGER DEFAULT 0,
                -- Extra JSON metadata
                extra           TEXT
            )
        """)
        await db.commit()
