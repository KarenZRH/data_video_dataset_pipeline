from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import ensure_dir


DDL = """
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sample_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  decision TEXT NOT NULL,
  original_value TEXT,
  reviewed_value TEXT,
  reviewer TEXT,
  reviewed_at TEXT NOT NULL,
  notes TEXT,
  model_version TEXT,
  config_hash TEXT
);
"""


def init_db(path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with sqlite3.connect(path) as conn:
        conn.execute(DDL)


def save_review(path: str | Path, row: dict[str, Any]) -> None:
    init_db(path)
    payload = {
        "sample_id": row.get("sample_id"),
        "stage": row.get("stage"),
        "decision": row.get("decision"),
        "original_value": json.dumps(row.get("original_value"), ensure_ascii=False),
        "reviewed_value": json.dumps(row.get("reviewed_value"), ensure_ascii=False),
        "reviewer": row.get("reviewer"),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "notes": row.get("notes"),
        "model_version": row.get("model_version"),
        "config_hash": row.get("config_hash"),
    }
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO reviews
            (sample_id, stage, decision, original_value, reviewed_value, reviewer, reviewed_at, notes, model_version, config_hash)
            VALUES (:sample_id, :stage, :decision, :original_value, :reviewed_value, :reviewer, :reviewed_at, :notes, :model_version, :config_hash)
            """,
            payload,
        )


def latest_reviews_by_clip(path: str | Path, sample_id: str, stage: str = "stage1_review") -> dict[str, dict[str, Any]]:
    init_db(path)
    rows: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT *
            FROM reviews
            WHERE sample_id = ? AND stage = ?
            ORDER BY reviewed_at ASC, id ASC
            """,
            (sample_id, stage),
        )
        for row in cursor:
            original_value = json.loads(row["original_value"] or "{}")
            reviewed_value = json.loads(row["reviewed_value"] or "{}")
            clip_id = (
                reviewed_value.get("clip_id")
                or reviewed_value.get("clip", {}).get("clip_id")
                or original_value.get("clip", {}).get("clip_id")
            )
            if not clip_id:
                continue
            rows[clip_id] = {
                "id": row["id"],
                "sample_id": row["sample_id"],
                "stage": row["stage"],
                "decision": row["decision"],
                "original_value": original_value,
                "reviewed_value": reviewed_value,
                "reviewer": row["reviewer"],
                "reviewed_at": row["reviewed_at"],
                "notes": row["notes"],
                "model_version": row["model_version"],
                "config_hash": row["config_hash"],
            }
    return rows
