"""Fixtures : corpus synthétique (documents de test, pas des extraits réels du BOFiP), encodeur sans modèle."""

from pathlib import Path

import pytest

from rag_fiscal.config import Settings
from rag_fiscal.pipeline import RAGPipeline

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(corpus_path=FIXTURES / "mini_corpus.jsonl", embedding_backend="hashing", reranker_backend="lexical",
                    llm_provider="none", qdrant_url=None, qdrant_path=":memory:", collection="test", top_k=3,
                    abstain_threshold=0.5, chunk_max_chars=400, api_keys="")


@pytest.fixture(scope="session")
def pipeline(settings: Settings) -> RAGPipeline:
    return RAGPipeline.from_settings(settings)
