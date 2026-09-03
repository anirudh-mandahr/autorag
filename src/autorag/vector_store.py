"""Vector store backends: pgvector (default) and FAISS fallback."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from autorag.config import PipelineConfig, Settings


@dataclass
class ScoredChunk:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float
    embedding: list[float] | None = None


class VectorBackend(ABC):
    @abstractmethod
    def reset(self, dimension: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, chunks: list[ScoredChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: list[float], k: int) -> list[ScoredChunk]:
        raise NotImplementedError


class PgVectorBackend(VectorBackend):
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._dimension = 0

    def reset(self, dimension: int) -> None:
        self._dimension = dimension
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute("DROP TABLE IF EXISTS autorag_chunks")
                cur.execute(
                    f"""
                    CREATE TABLE autorag_chunks (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({dimension}) NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX autorag_chunks_embedding_hnsw_idx
                    ON autorag_chunks
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
            conn.commit()

    def upsert(self, chunks: list[ScoredChunk]) -> None:
        if not chunks:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    if chunk.embedding is None:
                        raise ValueError(f"Chunk {chunk.id} is missing an embedding")
                    cur.execute(
                        """
                        INSERT INTO autorag_chunks (id, content, metadata, embedding)
                        VALUES (%s, %s, %s::jsonb, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        (
                            chunk.id,
                            chunk.content,
                            json.dumps(chunk.metadata),
                            chunk.embedding,
                        ),
                    )
            conn.commit()

    def search(self, query_embedding: list[float], k: int) -> list[ScoredChunk]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, metadata, embedding,
                           1 - (embedding <=> %s::vector) AS score
                    FROM autorag_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, k),
                )
                rows = cur.fetchall()

        return [
            ScoredChunk(
                id=row[0],
                content=row[1],
                metadata=row[2] if isinstance(row[2], dict) else json.loads(row[2]),
                embedding=list(row[3]),
                score=float(row[4]),
            )
            for row in rows
        ]

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._database_url)
        register_vector(conn)
        return conn


class FaissBackend(VectorBackend):
    def __init__(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "VECTOR_BACKEND=faiss requires the faiss extra: uv sync --extra faiss"
            ) from exc

        self._dimension = 0
        self._index: Any = None
        self._chunks: list[ScoredChunk] = []

    def reset(self, dimension: int) -> None:
        import faiss

        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks = []

    def upsert(self, chunks: list[ScoredChunk]) -> None:
        import faiss

        if not chunks:
            return
        if self._index is None:
            self.reset(len(chunks[0].embedding or []))

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} is missing an embedding")
            existing = next(
                (idx for idx, item in enumerate(self._chunks) if item.id == chunk.id), None
            )
            if existing is None:
                self._chunks.append(chunk)
            else:
                self._chunks[existing] = chunk

        self._index = faiss.IndexFlatIP(self._dimension)
        matrix = np.array([chunk.embedding for chunk in self._chunks], dtype=np.float32)
        self._index.add(matrix)

    def search(self, query_embedding: list[float], k: int) -> list[ScoredChunk]:
        if self._index is None or not self._chunks:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        scores, indices = self._index.search(query, min(k, len(self._chunks)))
        results: list[ScoredChunk] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0:
                continue
            chunk = self._chunks[int(index)]
            results.append(
                ScoredChunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    embedding=chunk.embedding,
                    score=float(score),
                )
            )
        return results


class VectorStore:
    """Abstracts pgvector + HNSW or local FAISS based on VECTOR_BACKEND."""

    def __init__(self, settings: Settings, config: PipelineConfig) -> None:
        self._settings = settings
        self._config = config
        backend_name = settings.vector_backend.lower()
        if backend_name == "faiss":
            self._backend: VectorBackend = FaissBackend()
        elif backend_name == "pgvector":
            self._backend = PgVectorBackend(settings.database_url)
        else:
            raise ValueError(f"Unknown VECTOR_BACKEND={settings.vector_backend!r}")

    def reset(self, dimension: int) -> None:
        self._backend.reset(dimension)

    def upsert(self, chunks: list[ScoredChunk]) -> None:
        self._backend.upsert(chunks)

    def search(self, query_embedding: list[float], k: int) -> list[ScoredChunk]:
        return self._backend.search(query_embedding, k)
