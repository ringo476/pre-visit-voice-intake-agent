"""Tests for the live-check harness itself, with a scripted stand-in for the
model — they prove the harness scores correctly, not that Gemini is accurate
(that is what `python -m app.eval.verifier_check` is for)."""

from langchain_core.messages import AIMessage

from app.eval.verifier_check import CASES, format_report, run_check


class _ByKeyword:
    """A pretend verifier that classifies by crude keywords, so the harness
    has realistic right and wrong answers to score."""

    def invoke(self, messages):
        reply = messages[0].content.split("What the patient said afterwards:")[1].split("\n")[0].lower()
        if any(w in reply for w in ["remember", "not sure", "no idea", "don't know"]):
            return AIMessage(content="UNSURE")
        if reply.strip(' ".').startswith(("no", "nope", "not at all", "i haven't")):
            return AIMessage(content="NEGATIVE")
        return AIMessage(content="OTHER")


class _AlwaysNegative:
    def invoke(self, messages):
        return AIMessage(content="NEGATIVE")


class _Broken:
    def invoke(self, messages):
        return AIMessage(content="no clue")


def test_every_case_has_a_valid_expected_label():
    assert {c.expected for c in CASES} <= {"negative", "unsure", "other"}
    assert {"negative", "unsure", "other"} == {c.expected for c in CASES}


def test_a_good_verifier_scores_perfectly_on_the_unambiguous_cases():
    results = run_check(_ByKeyword())
    assert all(r.passed for r in results), format_report(results)


def test_a_verifier_that_says_negative_to_everything_is_caught():
    results = run_check(_AlwaysNegative())
    failed = [r for r in results if not r.passed]
    assert any(r.case.expected == "other" for r in failed)
    assert any(r.case.expected == "unsure" for r in failed)


def test_unusable_answers_are_reported_as_errors_not_passes(monkeypatch):
    monkeypatch.setattr("app.retry.time.sleep", lambda s: None)
    results = run_check(_Broken())
    assert all(r.verdict is None and r.error for r in results)
    assert "ERROR" in format_report(results)
