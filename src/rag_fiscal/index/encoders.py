"""Encodeurs : vecteurs denses (sens) et creux (mots-clés, façon BM25).

- Dense : modèle sentence-transformers multilingue (multilingual-e5 par défaut), ou un encodeur par
  hachage de trigrammes, sans modèle, pour les tests et la CI.
- Creux : poids BM25 sur les termes ; l'IDF est appliqué par Qdrant (Modifier.IDF), qui le recalcule
  sur toute la collection. La recherche hybride fusionne ensuite les deux classements (RRF).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import numpy as np

STOPWORDS = set(
    "a au aux avec ce ces cet cette d dans de des du elle en est et il ils je l la le les leur lui m ma mais me "
    "meme mes moi mon n ne nos notre nous on ou par pas pour qu que qui s sa se ses son sont sur t ta te tes "
    "toi ton tu un une vos votre vous y quel quelle quels quelles comment est-ce etre avoir peut doit lorsque "
    "si ainsi cela ceci dont ou".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(normalize(text)) if t not in STOPWORDS and len(t) > 1]


def _stable_hash(token: str, buckets: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little") % buckets


class SparseEncoder:
    """Partie « fréquence » de BM25 (k1, b). L'IDF est délégué à Qdrant."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 180.0, buckets: int = 2**31 - 1):
        self.k1, self.b, self.avg_len, self.buckets = k1, b, avg_len, buckets

    def fit(self, texts: Sequence[str]) -> SparseEncoder:
        lengths = [len(tokenize(t)) for t in texts]
        self.avg_len = sum(lengths) / max(len(lengths), 1) or 1.0
        return self

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        tokens = tokenize(text)
        tf = Counter(tokens)
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_len)
        weights: dict[int, float] = {}
        for term, f in tf.items():
            idx = _stable_hash(term, self.buckets)
            weights[idx] = weights.get(idx, 0.0) + f * (self.k1 + 1) / (f + norm)
        return list(weights), list(weights.values())

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        idx = sorted({_stable_hash(t, self.buckets) for t in tokenize(text)})
        return idx, [1.0] * len(idx)


class DenseEncoder(Protocol):
    dim: int

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class HashingEncoder:
    """Encodeur dense sans modèle : trigrammes de caractères hachés. Utilisé pour les tests et la CI."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            padded = f"#{tok}#"
            for i in range(max(len(padded) - 2, 1)):
                vec[_stable_hash(padded[i:i + 3], self.dim)] += 1.0
        n = np.linalg.norm(vec)
        return vec / n if n else vec

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._encode(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(text)


class SentenceTransformerEncoder:
    """Modèle multilingual-e5 : les préfixes « query: » et « passage: » font partie de son entraînement."""

    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.e5 = "e5" in model_name.lower()

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        texts = [f"passage: {t}" for t in texts] if self.e5 else list(texts)
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)

    def encode_query(self, text: str) -> np.ndarray:
        text = f"query: {text}" if self.e5 else text
        return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]


def make_dense_encoder(backend: str, model_name: str) -> DenseEncoder:
    if backend == "hashing":
        return HashingEncoder()
    if backend == "sentence-transformers":
        return SentenceTransformerEncoder(model_name)
    raise ValueError(f"Encodeur inconnu : {backend}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


__all__ = ["SparseEncoder", "HashingEncoder", "SentenceTransformerEncoder", "make_dense_encoder", "tokenize",
           "normalize", "cosine"]
