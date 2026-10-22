"""Construit (ou reconstruit) l'index Qdrant à partir du corpus : découpage, embeddings, vecteurs creux."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_fiscal.config import get_settings  # noqa: E402
from rag_fiscal.index.encoders import SparseEncoder, make_dense_encoder  # noqa: E402
from rag_fiscal.index.store import VectorStore, make_client  # noqa: E402
from rag_fiscal.ingest.chunking import chunk_corpus, load_corpus  # noqa: E402

if __name__ == "__main__":
    s = get_settings()
    start = time.perf_counter()
    docs = load_corpus(s.corpus_path)
    chunks = list(chunk_corpus(docs, s.chunk_max_chars, s.chunk_overlap_paragraphs))
    store = VectorStore(make_client(s.qdrant_url, s.qdrant_path), s.collection,
                        make_dense_encoder(s.embedding_backend, s.embedding_model), SparseEncoder())
    store.recreate()
    n = store.index(chunks)
    print(f"{len(docs)} documents, {n} passages indexés en {time.perf_counter() - start:.0f} s")
