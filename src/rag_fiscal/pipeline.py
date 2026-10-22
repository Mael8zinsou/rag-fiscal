"""Chaîne complète : recherche hybride, reranking, génération. Chaque étape est mesurée et tracée."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from opentelemetry import trace
from prometheus_client import Counter, Histogram

from rag_fiscal.config import Settings
from rag_fiscal.generate.answer import Answer, answer_question, choose_prompt_version, load_prompt
from rag_fiscal.generate.llm import LLM, make_llm
from rag_fiscal.index.encoders import SparseEncoder, make_dense_encoder
from rag_fiscal.index.store import SearchFilters, VectorStore, make_client
from rag_fiscal.ingest.chunking import chunk_corpus, load_corpus
from rag_fiscal.retrieve.rerank import Reranker, make_reranker, rerank

tracer = trace.get_tracer("rag_fiscal")

STAGE_LATENCY = Histogram("rag_stage_seconds", "Durée de chaque étape", ["stage"],
                          buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10))
ANSWERS = Counter("rag_answers_total", "Réponses produites", ["mode", "prompt_version", "abstained"])
GROUNDING_ISSUES = Counter("rag_grounding_issues_total", "Réponses LLM rejetées par la vérification des citations")
TOP_SCORE = Histogram("rag_top_rerank_score", "Score du meilleur passage après reranking",
                      buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))


@contextmanager
def stage(name: str, timings: dict):
    start = time.perf_counter()
    with tracer.start_as_current_span(name):
        yield
    elapsed = time.perf_counter() - start
    STAGE_LATENCY.labels(name).observe(elapsed)
    timings[name] = round(elapsed * 1000, 1)


@dataclass
class RAGResult:
    answer: Answer
    timings_ms: dict

    def to_dict(self) -> dict:
        return {**asdict(self.answer), "timings_ms": self.timings_ms}


class RAGPipeline:
    def __init__(self, store: VectorStore, reranker: Reranker, llm: LLM | None, settings: Settings):
        self.store, self.reranker, self.llm, self.settings = store, reranker, llm, settings

    @classmethod
    def from_settings(cls, settings: Settings, build_index: bool = True, log=None) -> RAGPipeline:
        dense = make_dense_encoder(settings.embedding_backend, settings.embedding_model)
        store = VectorStore(make_client(settings.qdrant_url, settings.qdrant_path), settings.collection, dense, SparseEncoder())
        if build_index and (not store.client.collection_exists(settings.collection)
                            or store.client.count(settings.collection).count == 0):
            store.recreate()
            store.index(chunk_corpus(load_corpus(settings.corpus_path), settings.chunk_max_chars,
                                     settings.chunk_overlap_paragraphs), log=log)
        elif log:
            log(f"index existant réutilisé : {store.client.count(settings.collection).count} passages")
        reranker = make_reranker(settings.reranker_backend, settings.reranker_model)
        llm = make_llm(settings.llm_provider, settings.llm_model, settings.llm_api_key, settings.llm_base_url,
                       settings.llm_timeout_s)
        return cls(store, reranker, llm, settings)

    def retrieve(self, question: str, filters: SearchFilters | None = None, mode: str = "hybrid",
                 use_reranker: bool = True, timings: dict | None = None):
        timings = {} if timings is None else timings
        with stage("retrieve", timings):
            hits = self.store.search(question, mode=mode, limit=self.settings.prefetch_k, filters=filters)
        if use_reranker:
            with stage("rerank", timings):
                hits = rerank(self.reranker, question, hits, self.settings.top_k)
        else:
            hits = hits[: self.settings.top_k]
        return hits

    def ask(self, question: str, filters: SearchFilters | None = None, routing_key: str = "",
            prompt_version: str | None = None) -> RAGResult:
        timings: dict = {}
        with tracer.start_as_current_span("rag.ask") as span:
            hits = self.retrieve(question, filters, timings=timings)
            if hits:
                TOP_SCORE.observe(hits[0].score)
            version = prompt_version or choose_prompt_version(self.settings.prompt_ab, routing_key or question)
            prompt = load_prompt(version) if self.llm else None
            with stage("generate", timings):
                answer = answer_question(question, hits, self.llm, prompt, self.settings.abstain_threshold)
            span.set_attribute("rag.mode", answer.mode)
            span.set_attribute("rag.abstained", answer.abstained)
            span.set_attribute("rag.prompt_version", answer.prompt_version or "none")
        ANSWERS.labels(answer.mode, answer.prompt_version or "none", str(answer.abstained).lower()).inc()
        if answer.grounding_issue and not answer.grounding_issue.startswith("LLM indisponible"):
            GROUNDING_ISSUES.inc()
        return RAGResult(answer=answer, timings_ms=timings)
