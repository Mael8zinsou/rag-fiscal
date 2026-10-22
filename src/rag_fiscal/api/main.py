"""API FastAPI : /ask, /health, /metrics.

Sécurité : clé d'API par client (en-tête X-API-Key), associée à un tenant. Le tenant filtre les documents
visibles dans Qdrant, ce qui isole les clients les uns des autres. Une passerelle OIDC (Keycloak, par exemple)
peut se placer devant l'API et transmettre l'identité dans ce même en-tête.
"""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel, Field

from rag_fiscal.config import Settings, get_settings
from rag_fiscal.index.store import SearchFilters
from rag_fiscal.pipeline import RAGPipeline

REQUESTS = Counter("rag_http_requests_total", "Requêtes /ask", ["status"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    series: list[str] | None = Field(default=None, description="Séries du BOFiP, par exemple ['IFI', 'RPPM']")
    date_min: str | None = Field(default=None, description="Date minimale de publication, AAAA-MM-JJ")
    user_id: str | None = Field(default=None, description="Sert à l'affectation A/B des versions de prompt")


def parse_keys(spec: str) -> dict[str, str]:
    return dict(p.split(":", 1) for p in spec.split(",") if ":" in p)


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="RAG fiscal", version="0.1.0",
                  description="Questions-réponses sourcées sur la doctrine fiscale (BOFiP)")
    state: dict = {"pipeline": pipeline}
    keys = parse_keys(settings.api_keys)

    def get_pipeline() -> RAGPipeline:
        if state["pipeline"] is None:
            state["pipeline"] = RAGPipeline.from_settings(settings)
        return state["pipeline"]

    def get_tenant(x_api_key: str | None = Header(default=None)) -> str:
        if not keys:
            return "public"
        if x_api_key not in keys:
            REQUESTS.labels("401").inc()
            raise HTTPException(status_code=401, detail="Clé d'API absente ou invalide")
        return keys[x_api_key]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "llm": settings.llm_provider, "reranker": settings.reranker_backend,
                "embedding": settings.embedding_model if settings.embedding_backend != "hashing" else "hashing"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/ask")
    def ask(req: AskRequest, tenant: str = Depends(get_tenant), rag: RAGPipeline = Depends(get_pipeline)) -> dict:
        request_id = str(uuid.uuid4())
        filters = SearchFilters(tenant=tenant, series=req.series, date_min=req.date_min)
        result = rag.ask(req.question, filters, routing_key=req.user_id or request_id)
        REQUESTS.labels("200").inc()
        return {"request_id": request_id, "tenant": tenant, **result.to_dict()}

    _setup_tracing(app)
    return app


def _setup_tracing(app: FastAPI) -> None:
    """Active OpenTelemetry si OTEL_EXPORTER_OTLP_ENDPOINT est défini (Jaeger dans docker-compose)."""
    import os

    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "rag-fiscal"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)


app = create_app()
