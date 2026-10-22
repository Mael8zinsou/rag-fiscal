from rag_fiscal.index.store import SearchFilters
from rag_fiscal.retrieve.rerank import LexicalReranker


def top_doc(pipeline, question, **kw):
    hits = pipeline.retrieve(question, SearchFilters(**kw))
    return hits[0].payload["doc_id"] if hits else None


def test_hybrid_search_finds_the_right_document(pipeline):
    assert top_doc(pipeline, "seuil de patrimoine immobilier pour l'impôt sur la fortune immobilière") == "TEST-IFI-BASE"
    assert top_doc(pipeline, "option pour le barème progressif sur les dividendes") == "TEST-RPPM-PFU"
    assert top_doc(pipeline, "micro-foncier plafond de revenus bruts") == "TEST-RFPI-MICRO"


def test_every_mode_returns_results(pipeline):
    for mode in ("dense", "sparse", "hybrid"):
        assert pipeline.retrieve("abattement donation enfants", mode=mode)


def test_series_filter_restricts_results(pipeline):
    hits = pipeline.retrieve("abattement", SearchFilters(series=["RFPI"]))
    assert hits and {h.payload["serie"] for h in hits} == {"RFPI"}


def test_tenant_isolation(pipeline):
    question = "procédure de revue annuelle des dossiers du cabinet Alpha"
    public_docs = {h.payload["doc_id"] for h in pipeline.retrieve(question, SearchFilters(tenant="public"))}
    alpha_docs = {h.payload["doc_id"] for h in pipeline.retrieve(question, SearchFilters(tenant="cabinet-alpha"))}
    other_docs = {h.payload["doc_id"] for h in pipeline.retrieve(question, SearchFilters(tenant="cabinet-beta"))}
    assert "TEST-PRIVE-CABINET" in alpha_docs
    assert "TEST-PRIVE-CABINET" not in public_docs | other_docs


def test_lexical_reranker_prefers_passages_covering_the_question():
    scores = LexicalReranker().score("abattement assurance vie", ["abattement sur l'assurance vie", "régime micro-foncier"])
    assert scores[0] > scores[1]
