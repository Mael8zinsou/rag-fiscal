"""Reranking : on récupère large (30 passages) puis on reclasse finement les meilleurs.

Le score du reranker sert aussi de garde-fou : sous le seuil d'abstention, on ne répond pas.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from rag_fiscal.index.encoders import tokenize
from rag_fiscal.index.store import Hit


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class NoReranker:
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        # Conserve l'ordre de la recherche, avec un score décroissant dans [0, 1].
        n = len(texts)
        return [1.0 - i / max(n, 1) for i in range(n)]


class LexicalReranker:
    """Sans modèle : part des termes de la question présents dans le passage, pondérée par leur rareté."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        q_terms = set(tokenize(query))
        if not q_terms or not texts:
            return [0.0] * len(texts)
        doc_terms = [set(tokenize(t)) for t in texts]
        df = {t: sum(t in d for d in doc_terms) for t in q_terms}
        idf = {t: math.log(1 + len(texts) / (1 + df[t])) for t in q_terms}
        total = sum(idf.values()) or 1.0
        return [sum(idf[t] for t in q_terms if t in d) / total for d in doc_terms]


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=512)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        logits = self.model.predict([(query, t) for t in texts], show_progress_bar=False)
        return [1 / (1 + math.exp(-float(x))) for x in logits]   # sigmoïde : score lisible dans [0, 1]


def make_reranker(backend: str, model_name: str) -> Reranker:
    if backend == "none":
        return NoReranker()
    if backend == "lexical":
        return LexicalReranker()
    if backend == "cross-encoder":
        return CrossEncoderReranker(model_name)
    raise ValueError(f"Reranker inconnu : {backend}")


def rerank(reranker: Reranker, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    scores = reranker.score(query, [h.text for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    return [Hit(chunk_id=h.chunk_id, score=s, payload=h.payload) for h, s in ranked[:top_k]]
