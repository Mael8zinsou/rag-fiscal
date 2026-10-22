"""Construction de la réponse : prompt versionné, citations vérifiées, abstention, mode dégradé.

Garde-fous contre les hallucinations, dans l'ordre :
1. Abstention avant génération si aucun passage ne dépasse le seuil de pertinence.
2. Le LLM ne voit que les extraits numérotés et doit citer [n] après chaque affirmation.
3. Après génération, chaque citation est vérifiée : une référence inexistante, ou une réponse sans aucune
   citation, n'est pas renvoyée telle quelle. On bascule sur la réponse extractive, qui ne peut pas inventer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from rag_fiscal.config import ROOT
from rag_fiscal.generate.llm import LLM, LLMError
from rag_fiscal.index.store import Hit

ABSTAIN_MESSAGE = ("Je ne trouve pas, dans le BOFiP indexé, de passage suffisamment pertinent pour répondre "
                   "de façon fiable. Reformulez la question ou consultez un spécialiste.")
CITATION_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Citation:
    n: int
    doc_id: str
    title: str
    url: str
    excerpt: str
    score: float


@dataclass
class Answer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
    mode: str = "extractive"          # "llm", "extractive" (pas de LLM configuré) ou "degraded" (LLM en échec)
    prompt_version: str | None = None
    grounding_issue: str | None = None


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system: str
    user: str

    def render(self, question: str, context: str) -> tuple[str, str]:
        return self.system, self.user.format(question=question, context=context)


@lru_cache
def load_prompt(version: str, directory: str | None = None) -> PromptTemplate:
    path = Path(directory or ROOT / "prompts") / f"{version}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptTemplate(version=data["version"], system=data["system"], user=data["user"])


def parse_ab(spec: str) -> list[tuple[str, int]]:
    pairs = [(p.split(":")[0].strip(), int(p.split(":")[1])) for p in spec.split(",") if p.strip()]
    if sum(w for _, w in pairs) != 100:
        raise ValueError("La répartition A/B des prompts doit totaliser 100")
    return pairs


def choose_prompt_version(spec: str, routing_key: str) -> str:
    """Affectation déterministe : un même utilisateur garde la même version pendant tout le test."""
    bucket = int(hashlib.sha256(routing_key.encode()).hexdigest(), 16) % 100
    acc = 0
    for version, weight in parse_ab(spec):
        acc += weight
        if bucket < acc:
            return version
    return parse_ab(spec)[-1][0]


def _excerpt(text: str, n: int = 320) -> str:
    body = text.split("\n\n", 1)[-1]           # sans l'en-tête de titre
    return body if len(body) <= n else body[:n].rsplit(" ", 1)[0] + "…"


def to_citations(hits: list[Hit]) -> list[Citation]:
    return [Citation(n=i + 1, doc_id=h.payload["doc_id"], title=h.payload.get("title", ""), url=h.payload.get("url", ""),
                     excerpt=_excerpt(h.text), score=round(h.score, 4)) for i, h in enumerate(hits)]


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i + 1}] ({h.payload['doc_id']}) {h.text}" for i, h in enumerate(hits))


def extractive_answer(hits: list[Hit], max_passages: int = 3) -> str:
    lines = ["Passages du BOFiP les plus pertinents :"]
    for i, h in enumerate(hits[:max_passages]):
        lines.append(f"[{i + 1}] {_excerpt(h.text, 400)}")
    return "\n\n".join(lines)


def check_citations(text: str, n_sources: int) -> str | None:
    cited = {int(m) for m in CITATION_RE.findall(text)}
    if not cited:
        return "aucune citation"
    invalid = sorted(c for c in cited if c < 1 or c > n_sources)
    return f"citations inexistantes : {invalid}" if invalid else None


def answer_question(question: str, hits: list[Hit], llm: LLM | None, prompt: PromptTemplate | None,
                    abstain_threshold: float) -> Answer:
    if not hits or hits[0].score < abstain_threshold:
        return Answer(answer=ABSTAIN_MESSAGE, abstained=True, mode="llm" if llm else "extractive",
                      prompt_version=prompt.version if (llm and prompt) else None)
    citations = to_citations(hits)
    if llm is None or prompt is None:
        return Answer(answer=extractive_answer(hits), citations=citations, mode="extractive")

    system, user = prompt.render(question, build_context(hits))
    try:
        text = llm.complete(system, user)
    except LLMError as exc:
        return Answer(answer=extractive_answer(hits), citations=citations, mode="degraded",
                      prompt_version=prompt.version, grounding_issue=f"LLM indisponible : {exc}")
    if text.strip().upper().startswith("ABSTENTION"):
        return Answer(answer=ABSTAIN_MESSAGE, abstained=True, mode="llm", prompt_version=prompt.version)
    issue = check_citations(text, len(hits))
    if issue:
        return Answer(answer=extractive_answer(hits), citations=citations, mode="degraded",
                      prompt_version=prompt.version, grounding_issue=issue)
    used = {int(m) for m in CITATION_RE.findall(text)}
    return Answer(answer=text, citations=[c for c in citations if c.n in used], mode="llm", prompt_version=prompt.version)
