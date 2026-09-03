"""Tunable pipeline configuration."""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

RerankStrategy = Literal["none", "cosine", "mmr"]


class PipelineConfig(BaseModel):
    """All tunable RAG pipeline knobs live here."""

    chunk_size: int = Field(default=400, ge=50, description="Characters per chunk.")
    chunk_overlap: int = Field(default=80, ge=0, description="Overlap between chunks.")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model id.",
    )
    retrieval_k: int = Field(default=4, ge=1, description="Top-k chunks to retrieve.")
    rerank_strategy: RerankStrategy = Field(
        default="cosine",
        description="Post-retrieval reranking: none, cosine, or mmr.",
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR trade-off (1=relevance, 0=diversity).",
    )
    prompt_template_id: str = Field(
        default="grounded_v1",
        description="Prompt template key for grounded answers with citations.",
    )
    refusal_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Refuse when top retrieval cosine similarity is below this.",
    )


class Settings(BaseSettings):
    """Environment-backed runtime settings (not mutated by the research loop)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    database_url: str = Field(
        default="postgresql://autorag:autorag@localhost:5432/autorag",
        validation_alias="DATABASE_URL",
    )
    vector_backend: str = Field(default="pgvector", validation_alias="VECTOR_BACKEND")
    llm_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct",
        validation_alias="LLM_MODEL",
    )
    llm_provider: str = Field(default="openrouter", validation_alias="LLM_PROVIDER")
    max_experiments: int = Field(default=15, ge=1, validation_alias="MAX_EXPERIMENTS")
    max_usd_spend: float = Field(
        default=5.0,
        ge=0.0,
        validation_alias="MAX_USD_SPEND",
        description=(
            "Hard per-session USD cap. The loop reserves conservative max-cost "
            "estimates before researcher and eval calls and will not start work "
            "that cannot fit in the remaining budget."
        ),
    )


def get_search_space() -> dict:
    """Tunable dimensions exposed to the researcher agent."""
    return {
        "chunk_size": {"type": "integer", "minimum": 50, "maximum": 800},
        "chunk_overlap": {"type": "integer", "minimum": 0, "maximum": 300},
        "embedding_model": {
            "type": "enum",
            "choices": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ],
        },
        "retrieval_k": {"type": "integer", "minimum": 1, "maximum": 12},
        "rerank_strategy": {"type": "enum", "choices": ["none", "cosine", "mmr"]},
        "mmr_lambda": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "prompt_template_id": {
            "type": "enum",
            "choices": ["grounded_v1", "grounded_concise"],
        },
        "refusal_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }


QUALITY_WEIGHTS: dict[str, float] = {
    "answerable_recall": 0.10,
    "retrieval_hit_at_k": 0.15,
    "citation_precision": 0.10,
    "citation_recall": 0.10,
    "groundedness": 0.20,
    "answer_correctness": 0.20,
    "correct_refusal_rate": 0.15,
}

COST_WEIGHT: float = 50.0


def compute_objective(metrics: dict[str, float]) -> float:
    """Composite score: weighted quality minus weighted cost."""
    quality = sum(QUALITY_WEIGHTS[name] * metrics[name] for name in QUALITY_WEIGHTS)
    return quality - COST_WEIGHT * metrics["avg_cost_per_query"]
