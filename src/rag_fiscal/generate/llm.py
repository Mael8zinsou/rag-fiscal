"""Clients LLM : Gemini, ou toute API compatible OpenAI (OpenAI, Mistral, Ollama, vLLM...).

Aucun SDK : de simples appels HTTP, avec délai maximal et une nouvelle tentative. En cas d'échec, l'appelant
bascule en mode dégradé (réponse extractive) plutôt que de renvoyer une erreur.
"""

from __future__ import annotations

from typing import Protocol

import httpx


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class GeminiLLM:
    def __init__(self, model: str, api_key: str, timeout: float = 20.0):
        self.name, self.model, self.api_key, self.timeout = f"gemini:{model}", model, api_key, timeout

    def complete(self, system: str, user: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.0},
        }
        data = _post_with_retry(url, body, {"x-goog-api-key": self.api_key}, self.timeout)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Réponse Gemini inattendue : {data}") from exc


class OpenAICompatibleLLM:
    def __init__(self, model: str, api_key: str | None, base_url: str, timeout: float = 20.0):
        self.name, self.model, self.api_key, self.base_url, self.timeout = f"openai:{model}", model, api_key, base_url, timeout

    def complete(self, system: str, user: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {"model": self.model, "temperature": 0.0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        data = _post_with_retry(f"{self.base_url.rstrip('/')}/chat/completions", body, headers, self.timeout)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Réponse inattendue : {data}") from exc


def _post_with_retry(url: str, body: dict, headers: dict, timeout: float, attempts: int = 2) -> dict:
    last: Exception | None = None
    for _ in range(attempts):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                last = LLMError(f"HTTP {resp.status_code}")
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            last = exc
    raise LLMError(str(last))


def make_llm(provider: str, model: str, api_key: str | None, base_url: str, timeout: float) -> LLM | None:
    if provider == "none":
        return None
    if provider == "gemini":
        if not api_key:
            raise ValueError("RAG_LLM_API_KEY est requis pour Gemini")
        return GeminiLLM(model, api_key, timeout)
    if provider == "openai":
        return OpenAICompatibleLLM(model, api_key, base_url, timeout)
    raise ValueError(f"Fournisseur LLM inconnu : {provider}")
