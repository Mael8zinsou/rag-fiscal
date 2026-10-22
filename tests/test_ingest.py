from rag_fiscal.index.encoders import SparseEncoder, tokenize
from rag_fiscal.ingest.chunking import Document, chunk_document, paragraphs


def make_doc(n_paras: int, size: int = 120) -> Document:
    text = "\n\n".join(f"{(i + 1) * 10}\n" + ("mot " * (size // 4)).strip() for i in range(n_paras))
    return Document(id="BOI-TEST", title="Titre du document", serie="TEST", text=text)


def test_paragraphs_split_on_numbered_paragraphs():
    paras = paragraphs(" 10\n Premier alinéa\n suite du même alinéa.\n 20\n Second.\n\n Note isolée.")
    assert paras == ["10 Premier alinéa suite du même alinéa.", "20 Second.", "Note isolée."]


def test_chunks_respect_max_size_and_carry_title():
    chunks = chunk_document(make_doc(12), max_chars=400, overlap=1)
    assert len(chunks) > 1
    assert all(c.text.startswith("Titre du document") for c in chunks)
    assert all(len(c.text) <= 400 + len("Titre du document\n\n") for c in chunks)


def test_chunks_overlap_by_one_paragraph():
    chunks = chunk_document(make_doc(12), max_chars=400, overlap=1)
    first_body, second_body = chunks[0].text.split("\n\n")[1:], chunks[1].text.split("\n\n")[1:]
    assert first_body[-1] == second_body[0]


def test_chunk_ids_are_stable():
    assert [c.chunk_id for c in chunk_document(make_doc(6), 400)] == [c.chunk_id for c in chunk_document(make_doc(6), 400)]


def test_tokenize_removes_accents_and_stopwords():
    assert tokenize("L'impôt sur la fortune immobilière") == ["impot", "fortune", "immobiliere"]


def test_sparse_encoder_is_deterministic_and_saturates_frequency():
    enc = SparseEncoder().fit(["impot impot impot revenu", "revenu foncier"])
    idx1, val1 = enc.encode_document("impot impot impot revenu")
    idx2, val2 = enc.encode_document("impot impot impot revenu")
    assert (idx1, val1) == (idx2, val2)
    weights = dict(zip(idx1, val1))
    impot = weights[enc.encode_query("impot")[0][0]]
    assert impot < 3 * weights[enc.encode_query("revenu")[0][0]]   # saturation BM25 : pas linéaire
