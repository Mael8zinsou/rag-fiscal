from fastapi.testclient import TestClient

from rag_fiscal.api.main import create_app


def test_ask_returns_sourced_answer(settings, pipeline):
    client = TestClient(create_app(settings, pipeline))
    resp = client.post("/ask", json={"question": "abattement sur la résidence principale à l'IFI"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["citations"][0]["doc_id"] == "TEST-IFI-BASE"
    assert body["mode"] == "extractive" and not body["abstained"]
    assert set(body["timings_ms"]) == {"retrieve", "rerank", "generate"}


def test_out_of_scope_question_abstains(settings, pipeline):
    client = TestClient(create_app(settings, pipeline))
    body = client.post("/ask", json={"question": "Quelle est la recette de la tarte tatin ?"}).json()
    assert body["abstained"] and not body["citations"]


def test_api_key_and_tenant(settings, pipeline):
    secured = settings.model_copy(update={"api_keys": "cle-alpha:cabinet-alpha"})
    client = TestClient(create_app(secured, pipeline))
    assert client.post("/ask", json={"question": "revue annuelle"}).status_code == 401
    resp = client.post("/ask", json={"question": "procédure de revue annuelle des dossiers du cabinet Alpha"},
                       headers={"X-API-Key": "cle-alpha"})
    assert resp.json()["tenant"] == "cabinet-alpha"
    assert resp.json()["citations"][0]["doc_id"] == "TEST-PRIVE-CABINET"


def test_health_and_metrics(settings, pipeline):
    client = TestClient(create_app(settings, pipeline))
    assert client.get("/health").json()["status"] == "ok"
    client.post("/ask", json={"question": "micro-foncier"})
    assert "rag_answers_total" in client.get("/metrics").text
