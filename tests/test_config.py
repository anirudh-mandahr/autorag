"""PipelineConfig validation must reject out-of-range knobs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autorag.config import PipelineConfig


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_size", 49),
        ("chunk_size", 0),
        ("chunk_overlap", -1),
        ("retrieval_k", 0),
        ("mmr_lambda", -0.01),
        ("mmr_lambda", 1.01),
        ("refusal_threshold", -0.1),
        ("refusal_threshold", 1.5),
    ],
)
def test_pipeline_config_rejects_out_of_range(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(**{field: value})


def test_pipeline_config_rejects_invalid_rerank_strategy() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(rerank_strategy="invalid")  # type: ignore[arg-type]


def test_pipeline_config_accepts_boundary_values() -> None:
    cfg = PipelineConfig(
        chunk_size=50,
        chunk_overlap=0,
        retrieval_k=1,
        mmr_lambda=0.0,
        refusal_threshold=1.0,
    )
    assert cfg.chunk_size == 50
    assert cfg.refusal_threshold == 1.0


def test_quality_weights_sum_to_one() -> None:
    from autorag.config import QUALITY_WEIGHTS

    assert sum(QUALITY_WEIGHTS.values()) == pytest.approx(1.0)
