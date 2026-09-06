"""Runs the eval scenarios through the real agent graph, but with a
scripted stand-in for Gemini instead of the live model — same
ModelStep-style dependency-injection pattern proven in agent/graph.py and
tests/test_graph.py. Swapping the scripted LLM for the real one requires no
change to run_scenario/run_eval_suite's scoring logic."""

import sys
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import AIMessage

from app.agent.graph import run_agent_turn
from app.agent.session import create_session
from app.documents.document_store import add_document, create_uploaded_document
from app.eval.types import EvalScenario, ScenarioTurn
from app.protocol.registry import get_protocol
from app.schemas.intake_record import Source
from app.state_engine import get_current_facts, get_current_fact, get_missing_fields

EVIDENCE_REQUIRED_SOURCES = {Source.PATIENT_REPORTED, Source.ASKED_AND_DENIED, Source.DOCUMENT_SOURCED}


def _resolve_placeholders(args: dict, session) -> dict:
    """Scenario tool-call args may reference state produced earlier in the
    conversation via a small placeholder syntax, resolved against the live
    session right before dispatch — e.g. "$LAST_QUESTION_EVENT_ID" or
    "$LAST_FACT_ID:onset"."""
    resolved = {}
    for key, value in args.items():
        if value == "$LAST_QUESTION_EVENT_ID":
            resolved[key] = session.question_events[-1].id if session.question_events else None
        elif isinstance(value, str) and value.startswith("$LAST_FACT_ID:"):
            field_name = value.split(":", 1)[1]
            fact = get_current_fact(session.record, field_name)
            resolved[key] = fact.id if fact else None
        else:
            resolved[key] = value
    return resolved


class ScriptedTurnLLM:
    """Replays one scenario turn's scripted tool calls, then returns the
    scripted final text. A stand-in for the real Gemini-backed model."""

    def __init__(self, session, turn: ScenarioTurn):
        self.session = session
        self.turn = turn
        self.dispatched = False

    def invoke(self, messages):
        if not self.dispatched and self.turn.tool_calls:
            self.dispatched = True
            calls = [
                {"name": c["name"], "args": _resolve_placeholders(c["args"], self.session), "id": f"call-{i}"}
                for i, c in enumerate(self.turn.tool_calls)
            ]
            return AIMessage(content="", tool_calls=calls)
        return AIMessage(content=self.turn.final_text, tool_calls=[])


@dataclass
class ScenarioCheck:
    name: str
    passed: bool
    detail: Optional[str] = None


@dataclass
class ScenarioResult:
    scenario_id: str
    passed: bool
    checks: list[ScenarioCheck]
    metrics: dict = field(default_factory=dict)


def run_scenario(scenario: EvalScenario) -> ScenarioResult:
    protocol = get_protocol(scenario.protocol_id)
    session = create_session(f"eval-{scenario.id}", protocol)

    for seed in scenario.seed_documents:
        doc = create_uploaded_document(seed.filename, seed.mime_type, seed.text)
        session.documents.append(doc)
        add_document(session.session_id, doc)

    for turn in scenario.turns:
        run_agent_turn(session, turn.patient_utterance, llm=ScriptedTurnLLM(session, turn))

    current = get_current_facts(session.record)
    checks: list[ScenarioCheck] = []

    for f in scenario.expectations.fields_should_be_covered:
        fact = current.get(f)
        checks.append(ScenarioCheck(f"covered:{f}", fact is not None and fact.source != Source.NOT_ASKED, fact.value if fact else None))

    for f in scenario.expectations.fields_should_not_be_denied:
        fact = current.get(f)
        checks.append(ScenarioCheck(f"not-falsely-denied:{f}", fact is None or fact.source != Source.ASKED_AND_DENIED))

    for f in scenario.expectations.fields_should_remain_missing:
        still_missing = any(m.field == f for m in get_missing_fields(session.record, protocol))
        checks.append(ScenarioCheck(f"still-missing:{f}", still_missing))

    safety_triggered = any(e.triggered for e in session.safety_log)
    if scenario.expectations.expect_safety_trigger is not None:
        checks.append(ScenarioCheck("safety-trigger", safety_triggered == scenario.expectations.expect_safety_trigger))

    has_correction = any(f.supersedes for f in session.record.facts)
    if scenario.expectations.expect_correction is not None:
        checks.append(ScenarioCheck("correction-recorded", has_correction == scenario.expectations.expect_correction))

    if scenario.expectations.expect_document_sourced_field:
        f = scenario.expectations.expect_document_sourced_field
        fact = current.get(f)
        checks.append(
            ScenarioCheck(
                f"document-sourced:{f}",
                fact is not None and fact.source == Source.DOCUMENT_SOURCED and bool(fact.evidence_span),
                fact.evidence_span if fact else None,
            )
        )

    facts_needing_evidence = [f for f in session.record.facts if f.source in EVIDENCE_REQUIRED_SOURCES]
    evidence_linked = sum(1 for f in facts_needing_evidence if f.evidence_span)
    evidence_linkage_rate = 1.0 if not facts_needing_evidence else evidence_linked / len(facts_needing_evidence)
    checks.append(ScenarioCheck("evidence-linkage", evidence_linkage_rate == 1.0, f"{round(evidence_linkage_rate * 100)}%"))

    return ScenarioResult(
        scenario_id=scenario.id,
        passed=all(c.passed for c in checks),
        checks=checks,
        metrics={"safety_triggered": safety_triggered, "has_correction": has_correction, "evidence_linkage_rate": evidence_linkage_rate},
    )


@dataclass
class EvalSummary:
    total_scenarios: int
    passed_scenarios: int
    red_flag_recall: Optional[float]
    false_escalation_rate: Optional[float]
    evidence_linkage_rate: float


@dataclass
class EvalReport:
    scenario_results: list[ScenarioResult]
    summary: EvalSummary


def run_eval_suite(scenarios: list[EvalScenario]) -> EvalReport:
    results = [run_scenario(s) for s in scenarios]
    by_id = {r.scenario_id: r for r in results}

    expect_trigger = [s for s in scenarios if s.expectations.expect_safety_trigger is True]
    expect_no_trigger = [s for s in scenarios if s.expectations.expect_safety_trigger is False]

    red_flag_recall = (
        None
        if not expect_trigger
        else sum(1 for s in expect_trigger if by_id[s.id].metrics["safety_triggered"]) / len(expect_trigger)
    )
    false_escalation_rate = (
        None
        if not expect_no_trigger
        else sum(1 for s in expect_no_trigger if by_id[s.id].metrics["safety_triggered"]) / len(expect_no_trigger)
    )
    evidence_linkage_rate = sum(r.metrics["evidence_linkage_rate"] for r in results) / len(results)

    return EvalReport(
        scenario_results=results,
        summary=EvalSummary(
            total_scenarios=len(scenarios),
            passed_scenarios=sum(1 for r in results if r.passed),
            red_flag_recall=red_flag_recall,
            false_escalation_rate=false_escalation_rate,
            evidence_linkage_rate=evidence_linkage_rate,
        ),
    )


def _print_report(report: EvalReport) -> None:
    for result in report.scenario_results:
        print(f"\n{'PASS' if result.passed else 'FAIL'} — {result.scenario_id}")
        for check in result.checks:
            detail = f" ({check.detail})" if check.detail else ""
            print(f"  {'ok  ' if check.passed else 'FAIL'} {check.name}{detail}")

    s = report.summary
    print("\n--- summary ---")
    print(f"scenarios passed: {s.passed_scenarios}/{s.total_scenarios}")
    print(f"red-flag recall: {s.red_flag_recall}")
    print(f"false escalation rate: {s.false_escalation_rate}")
    print(f"evidence linkage rate: {round(s.evidence_linkage_rate * 100)}%")


if __name__ == "__main__":
    from app.eval.scenarios import ALL_SCENARIOS

    report = run_eval_suite(ALL_SCENARIOS)
    _print_report(report)
    sys.exit(0 if report.summary.passed_scenarios == report.summary.total_scenarios else 1)
