"""Configuration, lue dans les variables d'environnement (préfixe RAG_) ou un fichier .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Corpus et index
    corpus_path: Path = ROOT / "data" / "bofip_subset.jsonl.gz"
    qdrant_url: str | None = None          # None : Qdrant embarqué (fichier local ou mémoire)
    qdrant_path: str = ":memory:"
    collection: str = "bofip"
    chunk_max_chars: int = 1200
    chunk_overlap_paragraphs: int = 1

    # Recherche
    embedding_backend: str = "sentence-transformers"   # ou "hashing" (tests, sans modèle)
    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_backend: str = "cross-encoder"              # "cross-encoder", "lexical" ou "none"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    prefetch_k: int = 30
    top_k: int = 5
    abstain_threshold: float = 0.35   # score de reranking minimal pour répondre (calibré par l'évaluation)

    # Génération
    llm_provider: str = "none"         # "none" (mode extractif), "gemini" ou "openai" (API compatible)
    llm_model: str = "gemini-2.5-flash-lite"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout_s: float = 20.0
    prompt_ab: str = "v1:50,v2:50"     # répartition A/B des versions de prompt

    # Sécurité : "clé:tenant,clé2:tenant2". Vide : API ouverte, tenant « public ».
    api_keys: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
