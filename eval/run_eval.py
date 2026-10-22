"""Évaluation : qualité de la recherche, abstention, et ancrage des réponses générées.

Le jeu de questions (eval/questions.jsonl) associe chaque question aux identifiants BOI attendus. Un passage
compte comme pertinent si son identifiant commence par l'un des préfixes attendus (la nomenclature du BOFiP
est hiérarchique). Les questions « hors périmètre » doivent produire une abstention.

Usage : python eval/run_eval.py [--llm]   (avec --llm, la génération est évaluée si une clé est configurée)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_fiscal.config import Settings  # noqa: E402
from rag_fiscal.index.store import SearchFilters  # noqa: E402
from rag_fiscal.pipeline import RAGPipeline  # noqa: E402

CONFIGS = [("dense", False), ("sparse", False), ("hybrid", False), ("hybrid", True)]


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_relevant(doc_id: str, expected: list[str]) -> bool:
    return any(doc_id.startswith(e) for e in expected)


def retrieval_metrics(rag: RAGPipeline, questions: list[dict], mode: str, use_reranker: bool) -> dict:
    hits1, hits5, rr, lat = [], [], [], []
    for q in questions:
        start = time.perf_counter()
        hits = rag.retrieve(q["question"], SearchFilters(series=q.get("series")), mode=mode, use_reranker=use_reranker)
        lat.append((time.perf_counter() - start) * 1000)
        ranks = [i for i, h in enumerate(hits[:10]) if is_relevant(h.payload["doc_id"], q["expected"])]
        hits1.append(bool(ranks) and ranks[0] == 0)
        hits5.append(bool(ranks) and ranks[0] < 5)
        rr.append(1 / (ranks[0] + 1) if ranks else 0.0)
    return {"hit@1": statistics.mean(hits1), "hit@5": statistics.mean(hits5), "mrr@10": statistics.mean(rr),
            "latency_ms_p50": statistics.median(lat)}


def abstention_metrics(rag: RAGPipeline, in_scope: list[dict], out_scope: list[dict], threshold: float) -> dict:
    def top_score(question: str) -> float:
        hits = rag.retrieve(question)
        return hits[0].score if hits else 0.0

    in_scores = [top_score(q["question"]) for q in in_scope]
    out_scores = [top_score(q["question"]) for q in out_scope]
    # Seuil qui équilibre au mieux les deux erreurs (abstention à tort, réponse hors périmètre)
    candidates = sorted(set(in_scores + out_scores))
    def balanced(t: float) -> float:
        ok_in = statistics.mean(s >= t for s in in_scores)
        ok_out = statistics.mean(s < t for s in out_scores) if out_scores else 1.0
        return (ok_in + ok_out) / 2
    best = max(candidates, key=balanced) if candidates else threshold
    return {"threshold": threshold, "suggested_threshold": round(best, 3), "suggested_balanced_accuracy": balanced(best),
            "false_abstention_rate": statistics.mean(s < threshold for s in in_scores),
            "correct_abstention_rate": statistics.mean(s < threshold for s in out_scores) if out_scores else None,
            "in_scope_top_score_median": statistics.median(in_scores),
            "out_scope_top_score_median": statistics.median(out_scores) if out_scores else None}


def generation_metrics(rag: RAGPipeline, in_scope: list[dict], out_scope: list[dict], sample: int) -> dict:
    """Évalue la génération sur un échantillon : le quota gratuit de Gemini est de 20 requêtes par jour."""
    in_sample, out_sample = in_scope[:sample], out_scope[: max(1, sample // 3)]
    rows = {}
    for version in ("v1", "v2"):
        answers_in, answers_out = [], []
        for q in in_sample:
            answers_in.append((q, rag.ask(q["question"], prompt_version=version).answer))
            time.sleep(7)   # 10 requêtes par minute au maximum sur l'offre gratuite
        for q in out_sample:
            answers_out.append(rag.ask(q["question"], prompt_version=version).answer)
            time.sleep(7)
        answers = [a for _, a in answers_in]
        unavailable = [a for a in answers if (a.grounding_issue or "").startswith("LLM indisponible")]
        rejected = [a for a in answers if a.mode == "degraded" and a not in unavailable]
        rows[version] = {
            "questions": len(in_sample) + len(out_sample),
            "llm_unavailable": len(unavailable) / len(answers),
            "llm_answers": sum(a.mode == "llm" and not a.abstained for a in answers) / len(answers),
            "rejected_by_citation_check": len(rejected) / len(answers),
            "false_abstention_rate": statistics.mean(a.abstained for a in answers),
            "correct_abstention_rate": statistics.mean(a.abstained for a in answers_out) if answers_out else None,
            "answers_citing_expected_source": statistics.mean(
                a.mode == "llm" and not a.abstained and any(is_relevant(c.doc_id, q["expected"]) for c in a.citations)
                for q, a in answers_in),
        }
    return rows


def write_report(results: dict, path: Path) -> None:
    s = results["settings"]
    lines = ["# Résultats de l'évaluation", "",
             f"Corpus : {results['corpus']['documents']} documents BOFiP, {results['corpus']['chunks']} passages. "
             f"Questions : {results['questions']['in_scope']} dans le périmètre, "
             f"{results['questions']['out_of_scope']} hors périmètre.",
             f"Modèles : `{s['embedding_model']}` (embeddings), `{s['reranker_model']}` (reranking).", "",
             "## Recherche", "", "| Configuration | Hit@1 | Hit@5 | MRR@10 | Latence médiane |", "|---|---|---|---|---|"]
    for name, m in results["retrieval"].items():
        lines.append(f"| {name} | {m['hit@1']:.0%} | {m['hit@5']:.0%} | {m['mrr@10']:.2f} | {m['latency_ms_p50']:.0f} ms |")
    a = results["abstention"]
    lines += ["", "## Abstention (seuil sur le score de reranking)", "",
              f"- Seuil : {a['threshold']} (seuil qui équilibre le mieux les deux erreurs sur ce jeu : "
              f"{a['suggested_threshold']}, exactitude équilibrée {a['suggested_balanced_accuracy']:.0%})",
              f"- Questions du périmètre où le système s'abstient à tort : {a['false_abstention_rate']:.0%}",
              f"- Questions hors périmètre où il s'abstient, comme attendu : {a['correct_abstention_rate']:.0%}",
              f"- Score médian du meilleur passage : {a['in_scope_top_score_median']:.2f} dans le périmètre, "
              f"{a['out_scope_top_score_median']:.2f} hors périmètre"]
    if results.get("generation"):
        n = next(iter(results["generation"].values()))["questions"]
        lines += ["", f"## Génération (A/B des versions de prompt, échantillon de {n} questions par version)", "",
                  "| Prompt | Réponses LLM citées | Rejetées par la vérification des citations | LLM indisponible (quota) | "
                  "Abstention à tort | Abstention hors périmètre | Citent la source attendue |",
                  "|---|---|---|---|---|---|---|"]
        for v, m in results["generation"].items():
            lines.append(f"| {v} | {m['llm_answers']:.0%} | {m['rejected_by_citation_check']:.0%} | "
                         f"{m['llm_unavailable']:.0%} | {m['false_abstention_rate']:.0%} | "
                         f"{m['correct_abstention_rate']:.0%} | {m['answers_citing_expected_source']:.0%} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "results.json"))
    ap.add_argument("--report", default=str(ROOT / "eval" / "REPORT.md"))
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--llm-sample", type=int, default=6, help="questions du périmètre évaluées en génération, par prompt")
    args = ap.parse_args()

    settings = Settings()
    if not args.llm:
        settings = settings.model_copy(update={"llm_provider": "none"})
    rag = RAGPipeline.from_settings(settings)
    questions = load_questions(Path(args.questions))
    in_scope = [q for q in questions if q.get("expected")]
    out_scope = [q for q in questions if not q.get("expected")]

    results = {
        "settings": {k: str(v) for k, v in settings.model_dump().items() if k != "llm_api_key"},
        "corpus": {"documents": len({p.payload["doc_id"] for p in rag.store.client.scroll(
            settings.collection, limit=100000, with_payload=["doc_id"])[0]}),
                   "chunks": rag.store.client.count(settings.collection).count},
        "questions": {"in_scope": len(in_scope), "out_of_scope": len(out_scope)},
        "retrieval": {f"{m}{' + rerank' if r else ''}": retrieval_metrics(rag, in_scope, m, r) for m, r in CONFIGS},
        "abstention": abstention_metrics(rag, in_scope, out_scope, settings.abstain_threshold),
    }
    if args.llm and rag.llm is not None:
        results["generation"] = generation_metrics(rag, in_scope, out_scope, args.llm_sample)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(results, Path(args.report))
    print(Path(args.report).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
