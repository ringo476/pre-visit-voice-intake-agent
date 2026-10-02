"""Tests for the live-check harness itself, with scripted stand-ins for the
model — they prove the harness batches, parses and scores correctly, not that
Gemini is accurate (that is what `python -m app.eval.verifier_check` is for)."""

import re

from langchain_core.messages import AIMessage

from app.eval.verifier_check import CASES, format_report, run_check


class _Oracle:
    """Answers every claim with its known-correct verdict, found by matching the
    quoted words and the claim text inside the prompt it is sent."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        prompt = messages[0].content
        blocks = re.split(r"(?m)^Claim \d+\n", prompt)[1:]
        lines = []
        for i, block in enumerate(blocks, start=1):
            match = next(
                c for c in CASES if f'"{c.said}"' in block and c.claim().topic in block and _says(c, block)
            )
            lines.append(f"{i}: {match.expected.upper()}")
        return AIMessage(content="\n".join(lines))


def _says(case, block):
    from app.agent.answer_verifier import _describe_claim

    return _describe_claim(case.claim()) in block


class _AlwaysSupported:
    def invoke(self, messages):
        n = len(re.findall(r"(?m)^Claim \d+$", messages[0].content))
        return AIMessage(content="\n".join(f"{i}: SUPPORTED" for i in range(1, n + 1)))


class _Broken:
    def invoke(self, messages):
        return AIMessage(content="no clue")


def test_every_case_has_a_valid_expected_label_and_all_three_appear():
    assert {c.expected for c in CASES} == {"supported", "contradicted", "unrelated"}
    assert {c.polarity for c in CASES} == {"present", "absent", "unknown"}


def test_a_correct_verifier_scores_perfectly():
    results = run_check(_Oracle())
    assert all(r.passed for r in results), format_report(results)


def test_cases_are_sent_in_small_batches():
    oracle = _Oracle()
    run_check(oracle, batch_size=4)
    assert oracle.calls == -(-len(CASES) // 4)  # ceiling division


def test_a_verifier_that_supports_everything_is_caught():
    results = run_check(_AlwaysSupported())
    failed = [r for r in results if not r.passed]
    assert any(r.case.expected == "contradicted" for r in failed)
    assert any(r.case.expected == "unrelated" for r in failed)


def test_unusable_answers_are_reported_as_errors_not_passes(monkeypatch):
    monkeypatch.setattr("app.retry.time.sleep", lambda s: None)
    results = run_check(_Broken())
    assert all(r.verdict is None and r.error for r in results)
    assert "ERROR" in format_report(results)
