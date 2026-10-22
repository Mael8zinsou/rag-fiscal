"""Chargement du corpus et découpage en passages.

Les documents du BOFiP sont structurés en paragraphes numérotés (10, 20, 30...). On découpe donc sur les
paragraphes, sans jamais couper au milieu de l'un d'eux, avec un paragraphe de recouvrement entre deux
passages. Chaque passage porte le titre du document en en-tête : le passage reste compréhensible seul,
pour la recherche comme pour le LLM.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    serie: str
    text: str
    url: str = ""
    date: str = ""
    tenant: str = "public"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    serie: str
    text: str
    url: str = ""
    date: str = ""
    tenant: str = "public"
    position: int = 0
    payload_extra: dict = field(default_factory=dict)

    def payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id, "title": self.title, "serie": self.serie,
            "text": self.text, "url": self.url, "date": self.date, "tenant": self.tenant, "position": self.position,
            **self.payload_extra,
        }


def load_corpus(path: str | Path) -> list[Document]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    docs = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                docs.append(Document(id=d["id"], title=d.get("title", ""), serie=d.get("serie", ""), text=d["text"],
                                     url=d.get("url", ""), date=d.get("date", ""), tenant=d.get("tenant", "public")))
    return docs


_SPLIT_RE = re.compile(r"\n\s*\n|\n(?=\d{1,4}\s)")


def paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT_RE.split(text)]
    return [p for p in parts if p]


def chunk_document(doc: Document, max_chars: int = 1200, overlap: int = 1) -> list[Chunk]:
    paras = paragraphs(doc.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    header = f"{doc.title}\n\n" if doc.title else ""

    def flush() -> None:
        body = "\n\n".join(current)
        n = len(chunks)
        chunks.append(Chunk(chunk_id=f"{doc.id}#{n}", doc_id=doc.id, title=doc.title, serie=doc.serie,
                            text=header + body, url=doc.url, date=doc.date, tenant=doc.tenant, position=n))

    for para in paras:
        # Un paragraphe trop long est coupé sur les phrases.
        pieces = [para] if len(para) <= max_chars else re.split(r"(?<=[.;:])\s+", para)
        for piece in pieces:
            if current and len("\n\n".join(current + [piece])) > max_chars:
                flush()
                current = current[-overlap:] if overlap else []
                if current and len("\n\n".join(current + [piece])) > max_chars:
                    current = []
            current.append(piece)
    if current:
        flush()
    return chunks


def chunk_corpus(docs: Iterable[Document], max_chars: int = 1200, overlap: int = 1) -> Iterator[Chunk]:
    for doc in docs:
        yield from chunk_document(doc, max_chars, overlap)
