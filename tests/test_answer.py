from collections import Counter

import pytest

from rag_fiscal.generate.answer import (
    ABSTAIN_MESSAGE,
    answer_question,
    check_citations,
    choose_prompt_version,
    load_prompt,
    parse_ab,
)
from rag_fiscal.generate.llm import LLMError
from rag_fiscal.index.store import Hit


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str | Exception):
        self.reply = reply

    def complete(self, system: str, user: str) -> str:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


HITS = [Hit("A#0", 0.9, {"doc_id": "BOI-A", "text": "Titre A\n\nTexte A", "title": "Titre A", "url": "u"}),
        Hit("B#0", 0.7, {"doc_id": "BOI-B", "text": "Titre B\n\nTexte B", "title": "Titre B", "url": "u"})]
PROMPT = load_prompt("v2")


def test_abstains_when_best_passage_is_below_threshold():
    ans = answer_question("q", HITS, FakeLLM("Réponse [1]"), PROMPT, abstain_threshold=0.95)
    assert ans.abstained and ans.answer == ABSTAIN_MESSAGE and not ans.citations


def test_abstains_when_nothing_is_retrieved():
    assert answer_question("q", [], None, None, 0.1).abstained


def test_extractive_mode_without_llm_cites_passages():
    ans = answer_question("q", HITS, None, None, 0.1)
    assert ans.mode == "extractive" and "[1]" in ans.answer and len(ans.citations) == 2


def test_valid_llm_answer_keeps_only_cited_sources():
    ans = answer_question("q", HITS, FakeLLM("La règle s'applique [2]."), PROMPT, 0.1)
    assert ans.mode == "llm" and [c.doc_id for c in ans.citations] == ["BOI-B"]


@pytest.mark.parametrize("reply", ["Réponse sans aucune source.", "Réponse avec une source inventée [7]."])
def test_ungrounded_llm_answer_falls_back_to_extractive(reply):
    ans = answer_question("q", HITS, FakeLLM(reply), PROMPT, 0.1)
    assert ans.mode == "degraded" and ans.grounding_issue and "[1]" in ans.answer


def test_llm_abstention_is_respected():
    assert answer_question("q", HITS, FakeLLM("ABSTENTION"), PROMPT, 0.1).abstained


def test_llm_failure_degrades_instead_of_failing():
    ans = answer_question("q", HITS, FakeLLM(LLMError("timeout")), PROMPT, 0.1)
    assert ans.mode == "degraded" and ans.citations


def test_check_citations():
    assert check_citations("ok [1] [2]", 2) is None
    assert check_citations("rien", 2) == "aucune citation"
    assert "inexistantes" in check_citations("faux [3]", 2)


def test_ab_assignment_is_deterministic_and_balanced():
    assert choose_prompt_version("v1:50,v2:50", "client-42") == choose_prompt_version("v1:50,v2:50", "client-42")
    split = Counter(choose_prompt_version("v1:50,v2:50", f"user-{i}") for i in range(2000))
    assert 900 < split["v1"] < 1100
    assert choose_prompt_version("v1:0,v2:100", "x") == "v2"


def test_ab_spec_must_sum_to_100():
    with pytest.raises(ValueError):
        parse_ab("v1:60,v2:60")


def test_prompts_are_versioned():
    assert {load_prompt("v1").version, load_prompt("v2").version} == {"v1", "v2"}
