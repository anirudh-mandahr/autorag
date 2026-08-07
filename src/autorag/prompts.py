"""Grounded-answer prompt templates keyed by PipelineConfig.prompt_template_id."""

from __future__ import annotations

PROMPT_TEMPLATES: dict[str, str] = {
    "grounded_v1": """You answer questions using ONLY the provided context.
Cite supporting chunks by their source id in the JSON response.

Context:
{context}

Question: {question}

Respond with valid JSON only:
{{"answer": "<concise grounded answer>", "source_ids": ["<id>", ...]}}""",
    "grounded_concise": """Answer from context only. Be brief. Cite source ids.

Context:
{context}

Question: {question}

JSON only:
{{"answer": "<answer>", "source_ids": ["<id>"]}}""",
}


def get_prompt_template(template_id: str) -> str:
    if template_id not in PROMPT_TEMPLATES:
        known = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(f"Unknown prompt_template_id={template_id!r}. Known: {known}")
    return PROMPT_TEMPLATES[template_id]
