import os

os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

from app.eval.runner import run_eval_suite, run_scenario
from app.eval.scenarios import (
    ALL_SCENARIOS,
    correction_scenario,
    document_upload_scenario,
    incomplete_record_scenario,
    leg_injury_straightforward_scenario,
    safety_trigger_scenario,
    straightforward_scenario,
    uncertain_answer_scenario,
)


def test_straightforward_scenario_passes():
    assert run_scenario(straightforward_scenario).passed is True


def test_correction_scenario_preserves_original_and_records_correction():
    result = run_scenario(correction_scenario)
    assert result.passed is True
    assert result.metrics["has_correction"] is True


def test_safety_trigger_scenario_escalates():
    result = run_scenario(safety_trigger_scenario)
    assert result.passed is True
    assert result.metrics["safety_triggered"] is True


def test_uncertain_answer_scenario_never_records_a_denial():
    assert run_scenario(uncertain_answer_scenario).passed is True


def test_incomplete_record_scenario_leaves_fields_honestly_open():
    assert run_scenario(incomplete_record_scenario).passed is True


def test_document_upload_scenario_records_document_sourced_fact():
    assert run_scenario(document_upload_scenario).passed is True


def test_leg_injury_scenario_routes_to_the_correct_protocol():
    """Proves the multi-protocol fix: this scenario runs entirely on
    fields that don't exist in the respiratory checklist, against a
    different protocol selected by scenario.protocol_id."""
    result = run_scenario(leg_injury_straightforward_scenario)
    assert result.passed is True


def test_eval_suite_aggregate_metrics():
    report = run_eval_suite(ALL_SCENARIOS)
    assert report.summary.total_scenarios == len(ALL_SCENARIOS)
    assert report.summary.passed_scenarios == len(ALL_SCENARIOS)
    assert report.summary.red_flag_recall == 1
    assert report.summary.false_escalation_rate == 0
    assert report.summary.evidence_linkage_rate == 1
