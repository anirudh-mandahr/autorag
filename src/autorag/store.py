"""SQLite persistence for research experiments."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from autorag.config import PipelineConfig
from autorag.eval import config_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / ".cache" / "research" / "experiments.db"


@dataclass(frozen=True)
class ExperimentRecord:
    """One row in the experiment history."""

    id: int
    config: PipelineConfig
    metrics: dict[str, float]
    objective: float
    status: str  # "kept" | "discarded"
    parent_id: int | None
    cost_usd: float
    is_baseline: bool
    created_at: str

    def to_summary(self) -> dict:
        return {
            "id": self.id,
            "config": self.config.model_dump(),
            "metrics": self.metrics,
            "objective": round(self.objective, 6),
            "status": self.status,
            "parent_id": self.parent_id,
            "cost_usd": round(self.cost_usd, 6),
            "is_baseline": self.is_baseline,
            "created_at": self.created_at,
        }


class ExperimentStore:
    """Persists experiment configs, metrics, and keep/discard decisions."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_json TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                objective REAL NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('kept', 'discarded')),
                parent_id INTEGER REFERENCES experiments(id),
                cost_usd REAL NOT NULL,
                is_baseline INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_experiments_config_hash
                ON experiments(config_hash);
            CREATE INDEX IF NOT EXISTS idx_experiments_status
                ON experiments(status);
            """
        )
        self._conn.commit()

    @staticmethod
    def hash_config(config: PipelineConfig) -> str:
        return config_hash(config)

    def has_config(self, config: PipelineConfig) -> bool:
        digest = self.hash_config(config)
        row = self._conn.execute(
            "SELECT 1 FROM experiments WHERE config_hash = ? LIMIT 1",
            (digest,),
        ).fetchone()
        return row is not None

    def seen_config_hashes(self) -> set[str]:
        rows = self._conn.execute("SELECT config_hash FROM experiments").fetchall()
        return {str(row["config_hash"]) for row in rows}

    def history(self, *, limit: int | None = None) -> list[ExperimentRecord]:
        query = "SELECT * FROM experiments ORDER BY id ASC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = self._conn.execute(query).fetchall()
        return [self._row_to_record(row) for row in rows]

    def recent(self, limit: int = 10) -> list[ExperimentRecord]:
        rows = self._conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_record(row) for row in reversed(rows)]

    def best(self) -> ExperimentRecord | None:
        row = self._conn.execute(
            """
            SELECT * FROM experiments
            WHERE status = 'kept'
            ORDER BY objective DESC, id ASC
            LIMIT 1
            """
        ).fetchone()
        return self._row_to_record(row) if row else None

    def baseline(self) -> ExperimentRecord | None:
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE is_baseline = 1 ORDER BY id ASC LIMIT 1"
        ).fetchone()
        return self._row_to_record(row) if row else None

    def total_cost_usd(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM experiments"
        ).fetchone()
        return float(row["total"]) if row else 0.0

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM experiments").fetchone()
        return int(row["n"])

    def insert(
        self,
        *,
        config: PipelineConfig,
        metrics: dict[str, float],
        objective: float,
        status: str,
        parent_id: int | None,
        cost_usd: float,
        is_baseline: bool = False,
    ) -> ExperimentRecord:
        created_at = datetime.now(UTC).isoformat()
        digest = self.hash_config(config)
        cursor = self._conn.execute(
            """
            INSERT INTO experiments (
                config_json, config_hash, metrics_json, objective,
                status, parent_id, cost_usd, is_baseline, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                json.dumps(config.model_dump(), sort_keys=True),
                digest,
                json.dumps(metrics, sort_keys=True),
                objective,
                status,
                parent_id,
                cost_usd,
                int(is_baseline),
                created_at,
            ),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to read back inserted experiment")
        return self._row_to_record(row)

    def _row_to_record(self, row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            id=int(row["id"]),
            config=PipelineConfig.model_validate(json.loads(row["config_json"])),
            metrics={k: float(v) for k, v in json.loads(row["metrics_json"]).items()},
            objective=float(row["objective"]),
            status=str(row["status"]),
            parent_id=int(row["parent_id"]) if row["parent_id"] is not None else None,
            cost_usd=float(row["cost_usd"]),
            is_baseline=bool(row["is_baseline"]),
            created_at=str(row["created_at"]),
        )
