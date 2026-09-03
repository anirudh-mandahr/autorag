"""LangChain LCEL RAG pipeline wired to PipelineConfig."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from autorag.config import PipelineConfig, Settings
from autorag.embeddings import EmbeddingClient
from autorag.llm import LLMClient
from autorag.prompts import get_prompt_template
from autorag.usage import UsageTracker
from autorag.vector_store import ScoredChunk, VectorStore

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class QueryResult(BaseModel):
    question: str
    answer: str
    source_ids: list[str] = Field(default_factory=list)
    refused: bool = False
    top_similarity: float = 0.0
    retrieved_ids: list[str] = Field(default_factory=list)
    context: str = ""


class RAGPipeline:
    """End-to-end retrieval-augmented generation chain."""

    def __init__(
        self,
        settings: Settings,
        config: PipelineConfig,
        llm: LLMClient | None = None,
        embeddings: EmbeddingClient | None = None,
        store: VectorStore | None = None,
        usage: UsageTracker | None = None,
    ) -> None:
        self._settings = settings
        self._config = config
        self.usage = usage or UsageTracker()
        self._llm = llm or LLMClient(settings, config, self.usage)
        self._embeddings = embeddings or EmbeddingClient(settings, config, self.usage)
        self._store = store or VectorStore(settings, config)
        self._indexed = False
        self._query_chain = self._build_query_chain()

    def index_corpus(self, text: str, *, source: str = "corpus") -> int:
        """Chunk, embed, and upsert a corpus into the vector store."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        chunks = splitter.split_text(text)
        if not chunks:
            self._store.reset(self._embeddings.dimension)
            self._indexed = True
            return 0

        embeddings = self._embeddings.embed_documents(chunks)
        self._store.reset(self._embeddings.dimension)
        scored_chunks = [
            ScoredChunk(
                id=f"{source}-chunk-{index}",
                content=content,
                metadata={"source": source, "chunk_index": index},
                score=0.0,
                embedding=embedding,
            )
            for index, (content, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        self._store.upsert(scored_chunks)
        self._indexed = True
        return len(scored_chunks)

    def query(self, question: str) -> QueryResult:
        """Run the LCEL query chain for a single question."""
        if not self._indexed:
            raise RuntimeError("Call index_corpus() before query()")
        payload: dict[str, Any] = self._query_chain.invoke({"question": question})
        return QueryResult.model_validate(payload)

    @property
    def metrics(self) -> dict[str, float | int]:
        return self.usage.summary()

    def _build_query_chain(self):
        embed = RunnableLambda(self._embed_question)
        retrieve = RunnableLambda(self._retrieve_chunks)
        annotate = RunnableLambda(self._annotate_similarity)
        refuse = RunnableLambda(self._refuse_answer)
        generate = RunnableLambda(self._generate_answer)

        return (
            RunnablePassthrough()
            | embed
            | retrieve
            | annotate
            | RunnableBranch(
                (lambda state: state["top_similarity"] < self._config.refusal_threshold, refuse),
                generate,
            )
        )

    def _embed_question(self, state: dict[str, Any]) -> dict[str, Any]:
        state["query_embedding"] = self._embeddings.embed_query(state["question"])
        return state

    def _retrieve_chunks(self, state: dict[str, Any]) -> dict[str, Any]:
        state["retrieved"] = self._store.search(
            state["query_embedding"],
            self._config.retrieval_k,
        )
        return state

    def _annotate_similarity(self, state: dict[str, Any]) -> dict[str, Any]:
        retrieved: list[ScoredChunk] = state.get("retrieved", [])
        state["top_similarity"] = retrieved[0].score if retrieved else 0.0
        state["retrieved_ids"] = [chunk.id for chunk in retrieved]
        return state

    def _refuse_answer(self, state: dict[str, Any]) -> dict[str, Any]:
        retrieved: list[ScoredChunk] = state.get("retrieved", [])
        context = "\n\n".join(f"[{chunk.id}] {chunk.content}" for chunk in retrieved)
        return {
            "question": state["question"],
            "answer": (
                "I don't have enough relevant information in the corpus to answer "
                "that question confidently."
            ),
            "source_ids": [],
            "refused": True,
            "top_similarity": state["top_similarity"],
            "retrieved_ids": state.get("retrieved_ids", []),
            "context": context,
        }

    def _generate_answer(self, state: dict[str, Any]) -> dict[str, Any]:
        reranked = self._rerank(state["query_embedding"], state["retrieved"])
        context = "\n\n".join(f"[{chunk.id}] {chunk.content}" for chunk in reranked)
        template = get_prompt_template(self._config.prompt_template_id)
        prompt = template.format(context=context, question=state["question"])
        answer, source_ids = self._llm.grounded_answer(prompt=prompt)

        valid_ids = {chunk.id for chunk in reranked}
        # Only model-emitted citations. Do not auto-cite the top chunk: a
        # retrieved-but-uncited source must not count as citation success.
        cited = [source_id for source_id in source_ids if source_id in valid_ids]

        return {
            "question": state["question"],
            "answer": answer,
            "source_ids": cited,
            "refused": False,
            "top_similarity": state["top_similarity"],
            "retrieved_ids": state.get("retrieved_ids", []),
            "context": context,
        }

    def _rerank(
        self,
        query_embedding: list[float],
        chunks: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        if not chunks:
            return []

        strategy = self._config.rerank_strategy
        if strategy == "none":
            return chunks

        if strategy == "cosine":
            return sorted(
                chunks,
                key=lambda chunk: self._cosine(query_embedding, chunk.embedding or []),
                reverse=True,
            )

        return self._mmr_rerank(query_embedding, chunks)

    def _mmr_rerank(
        self,
        query_embedding: list[float],
        chunks: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        lambda_mult = self._config.mmr_lambda
        selected: list[ScoredChunk] = []
        candidates = list(chunks)

        while candidates and len(selected) < self._config.retrieval_k:
            best_score = float("-inf")
            best_index = 0
            for index, candidate in enumerate(candidates):
                relevance = self._cosine(query_embedding, candidate.embedding or [])
                diversity = 0.0
                if selected:
                    diversity = max(
                        self._cosine(candidate.embedding or [], chosen.embedding or [])
                        for chosen in selected
                    )
                score = lambda_mult * relevance - (1.0 - lambda_mult) * diversity
                if score > best_score:
                    best_score = score
                    best_index = index
            selected.append(candidates.pop(best_index))

        return selected

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        left_arr = np.array(left, dtype=float)
        right_arr = np.array(right, dtype=float)
        denom = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
        if denom == 0.0:
            return 0.0
        return float(np.dot(left_arr, right_arr) / denom)


def load_sample_corpus(path: Path | None = None) -> str:
    corpus_path = path or (DATA_DIR / "corpus.txt")
    return corpus_path.read_text(encoding="utf-8")


def load_sample_questions(path: Path | None = None) -> list[dict[str, str]]:
    import json

    questions_path = path or (DATA_DIR / "questions.json")
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    return list(data)
