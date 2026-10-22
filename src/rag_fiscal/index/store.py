"""Index Qdrant : un vecteur dense et un vecteur creux par passage, recherche hybride fusionnée par RRF."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from rag_fiscal.index.encoders import DenseEncoder, SparseEncoder
from rag_fiscal.ingest.chunking import Chunk

DENSE, SPARSE = "dense", "sparse"


@dataclass
class Hit:
    chunk_id: str
    score: float
    payload: dict

    @property
    def text(self) -> str:
        return self.payload.get("text", "")


@dataclass
class SearchFilters:
    tenant: str = "public"
    series: Sequence[str] | None = None
    date_min: str | None = None   # AAAA-MM-JJ

    def to_qdrant(self) -> qm.Filter:
        must: list[qm.Condition] = [
            # Multi-tenant : un client ne voit que les documents publics et les siens.
            qm.FieldCondition(key="tenant", match=qm.MatchAny(any=sorted({"public", self.tenant})))
        ]
        if self.series:
            must.append(qm.FieldCondition(key="serie", match=qm.MatchAny(any=list(self.series))))
        if self.date_min:
            must.append(qm.FieldCondition(key="date", range=qm.DatetimeRange(gte=self.date_min)))
        return qm.Filter(must=must)


def make_client(url: str | None, path: str = ":memory:") -> QdrantClient:
    if url:
        return QdrantClient(url=url)
    return QdrantClient(location=":memory:") if path == ":memory:" else QdrantClient(path=path)


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dense: DenseEncoder, sparse: SparseEncoder):
        self.client, self.collection, self.dense, self.sparse = client, collection, dense, sparse

    def recreate(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: qm.VectorParams(size=self.dense.dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
        for key, schema in (("tenant", qm.PayloadSchemaType.KEYWORD), ("serie", qm.PayloadSchemaType.KEYWORD),
                            ("doc_id", qm.PayloadSchemaType.KEYWORD), ("date", qm.PayloadSchemaType.DATETIME)):
            try:
                self.client.create_payload_index(self.collection, key, schema)
            except Exception:  # noqa: BLE001 - le mode embarqué ignore les index de payload
                pass

    def index(self, chunks: Iterable[Chunk], batch_size: int = 64) -> int:
        chunks = list(chunks)
        self.sparse.fit([c.text for c in chunks])
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            dense_vecs = self.dense.encode_documents([c.text for c in batch])
            points = []
            for chunk, vec in zip(batch, dense_vecs):
                idx, val = self.sparse.encode_document(chunk.text)
                points.append(qm.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                    vector={DENSE: vec.tolist(), SPARSE: qm.SparseVector(indices=idx, values=val)},
                    payload={**chunk.payload(), "date": chunk.date or None},
                ))
            self.client.upsert(self.collection, points)
            total += len(points)
        return total

    def search(self, query: str, mode: str = "hybrid", limit: int = 30,
               filters: SearchFilters | None = None) -> list[Hit]:
        flt = (filters or SearchFilters()).to_qdrant()
        dense_q = self.dense.encode_query(query).tolist()
        s_idx, s_val = self.sparse.encode_query(query)
        sparse_q = qm.SparseVector(indices=s_idx, values=s_val)
        if mode == "dense":
            res = self.client.query_points(self.collection, query=dense_q, using=DENSE, query_filter=flt,
                                           limit=limit, with_payload=True)
        elif mode == "sparse":
            if not s_idx:
                return []
            res = self.client.query_points(self.collection, query=sparse_q, using=SPARSE, query_filter=flt,
                                           limit=limit, with_payload=True)
        elif mode == "hybrid":
            prefetch = [qm.Prefetch(query=dense_q, using=DENSE, filter=flt, limit=limit)]
            if s_idx:
                prefetch.append(qm.Prefetch(query=sparse_q, using=SPARSE, filter=flt, limit=limit))
            res = self.client.query_points(self.collection, prefetch=prefetch, query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                                           query_filter=flt, limit=limit, with_payload=True)
        else:
            raise ValueError(f"Mode de recherche inconnu : {mode}")
        return [Hit(chunk_id=p.payload["chunk_id"], score=float(p.score), payload=p.payload) for p in res.points]
