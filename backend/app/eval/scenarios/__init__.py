from app.eval.scenarios.straightforward import straightforward_scenario
from app.eval.scenarios.correction import correction_scenario
from app.eval.scenarios.safety_trigger import safety_trigger_scenario
from app.eval.scenarios.uncertain_answer import uncertain_answer_scenario
from app.eval.scenarios.incomplete_record import incomplete_record_scenario
from app.eval.scenarios.document_upload import document_upload_scenario
from app.eval.scenarios.leg_injury_straightforward import leg_injury_straightforward_scenario

ALL_SCENARIOS = [
    straightforward_scenario,
    correction_scenario,
    safety_trigger_scenario,
    uncertain_answer_scenario,
    incomplete_record_scenario,
    document_upload_scenario,
    leg_injury_straightforward_scenario,
]
