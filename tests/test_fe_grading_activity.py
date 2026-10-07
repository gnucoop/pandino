"""Tests for evaluation.datachat.grading.activity: the observed-activity summary.

Offline only: synthetic run packages built with the grading-test builders.
"""

import pytest

from evaluation.datachat.grading import activity as act
from evaluation.datachat.grading.model import ActivitySource, Gate, ObservedActivity, ScenarioGrade
from tests.test_fe_grading_grader import (  # noqa: F401
    FE01, agent, body, grade, loaded, repo_root, runtime, scenario_json,
)
from tests.test_fe_grading_catalogue import text
from tests.test_fe_grading_reader import USAGE, build_run

RT, CODE, USG = ActivitySource.RUNTIME, ActivitySource.GENERATED_CODE, ActivitySource.USAGE
NO_EXTRA = {**USAGE, "rows": [{"id": 7, "service": "/datachat"}]}


def summary(tmp_path, repo_root, **spec):
    return act.observed_activity(loaded(tmp_path, repo_root, "FE-01", **spec))


def rows(items):
    return [(a.source, a.name, a.count, a.outcome) for a in items]


def test_runtime_outcomes_aggregate_per_tool_and_outcome(tmp_path, repo_root):
    log = runtime(
        ("aggregate", "event=tool_call_result group_by=region"),
        ("aggregate", "event=tool_call_result group_by=channel"),
        ("classify_match", "event=tool_call_rejected tool=classify_match code=X"),
        ("classify_match", "event=tool_call_result tool=classify_match rows=400"),
        ("keywords", "event=tool_call_empty col=comment"),
        ("sentiment", "event=tool_call_failed tool=sentiment_analysis"),
        ("plot", "event=tool_call_result plot=/abs/secret/path.png"),
    )
    got = summary(tmp_path, repo_root, runtime_log=log, agent_record=agent("x = 1"), usage=NO_EXTRA)
    assert rows(got) == [
        (RT, "aggregate", 2, "result"),
        (RT, "classify_match", 1, "rejected"),
        (RT, "classify_match", 1, "result"),
        (RT, "keywords", 1, "empty"),
        (RT, "plot", 1, "result"),
        (RT, "sentiment_analysis", 1, "failed"),
    ]
    # Logged fields (paths, values) are never persisted.
    assert "secret" not in str([a.to_dict() for a in got])


def test_generated_code_is_inferred_counted_and_never_carries_an_outcome(tmp_path, repo_root):
    code = agent(
        "a = trend(date_col='d')\nb = trend(date_col='d', op='mean')",
        "df.plot(kind='bar')\nmy_trend(1)\nchart(kind='line', data=a)",
    )
    got = summary(tmp_path, repo_root, runtime_log=runtime(), agent_record=code, usage=NO_EXTRA)
    assert rows(got) == [(CODE, "chart", 1, None), (CODE, "trend", 2, None)]
    assert all(a.outcome is None for a in got)


def test_usage_only_is_provider_backed_and_never_names_a_tool(tmp_path, repo_root):
    usage = {**USAGE, "rows": [{"id": 5}, {"id": 6}, {"id": 7}]}
    got = summary(tmp_path, repo_root, runtime_log=runtime(), agent_record=agent("x = 1"), usage=usage)
    assert rows(got) == [(USG, act.PROVIDER_BACKED_CALL, 2, None)]
    assert act.PROVIDER_BACKED_CALL not in act.DATACHAT_TOOLS


def test_usage_without_known_primary_row_is_not_reported(tmp_path, repo_root):
    usage = {**USAGE, "primary_log_id": None}
    assert summary(tmp_path, repo_root, runtime_log=runtime(), agent_record=agent("x = 1"), usage=usage) == ()


def test_sources_stay_separate_for_the_same_tool(tmp_path, repo_root):
    log = runtime(("sentiment", "event=tool_call_result tool=sentiment_analysis"))
    got = summary(tmp_path, repo_root, runtime_log=log,
                  agent_record=agent("s = sentiment_analysis(col='comment')"), usage=USAGE)
    assert rows(got) == [
        (RT, "sentiment_analysis", 1, "result"),
        (CODE, "sentiment_analysis", 1, None),
        (USG, act.PROVIDER_BACKED_CALL, 1, None),
    ]


def test_no_evidence_gives_an_empty_summary(tmp_path, repo_root):
    assert summary(tmp_path, repo_root, runtime_log=runtime(), agent_record=None, usage=NO_EXTRA) == ()


def test_summary_is_deterministic_and_persisted(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01})
    grade(root, repo_root)
    first = scenario_json(root, "FE-01")
    grade(root, repo_root)
    assert scenario_json(root, "FE-01") == first
    assert first["activity"] == [
        {"name": "chart", "source": "RUNTIME", "count": 1, "outcome": "result"},
        {"name": "aggregate", "source": "GENERATED_CODE", "count": 1, "outcome": None},
        {"name": "chart", "source": "GENERATED_CODE", "count": 1, "outcome": None},
        {"name": act.PROVIDER_BACKED_CALL, "source": "USAGE", "count": 1, "outcome": None},
    ]


def test_activity_does_not_change_the_outcome(tmp_path, repo_root):
    spec = dict(body=body(text("ok")), runtime_log=runtime(), agent_record=agent("x = 1"))
    root = build_run(tmp_path, repo_root, {"FE-02": dict(spec, usage=NO_EXTRA), "FE-03": dict(spec, usage=USAGE)})
    grade(root, repo_root)
    a, b = scenario_json(root, "FE-02"), scenario_json(root, "FE-03")
    assert a["activity"] == [] and b["activity"]
    assert (a["outcome"], a["failures"]) == (b["outcome"], b["failures"])


def test_non_gradeable_scenarios_carry_no_activity(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": dict(scoreable=False)})
    grade(root, repo_root)
    assert scenario_json(root, "FE-01")["activity"] == []
    with pytest.raises(ValueError, match="no activity"):
        ScenarioGrade("FE-01", Gate.NOT_RUN, activity=(ObservedActivity("chart", CODE, 1),))


@pytest.mark.parametrize("kwargs", [
    dict(name="chart", source=CODE, count=1, outcome="result"),  # inferred code claims no outcome
    dict(name="x", source=USG, count=1, outcome="result"),
    dict(name="chart", source=RT, count=1, outcome=None),
    dict(name="chart", source=RT, count=0, outcome="result"),
    dict(name="", source=CODE, count=1),
    dict(name="chart", source="RUNTIME", count=1, outcome="result"),
])
def test_invalid_activity_refused(kwargs):
    with pytest.raises(ValueError):
        ObservedActivity(**kwargs)


def test_unaggregated_activity_refused():
    dup = ObservedActivity("chart", CODE, 1)
    with pytest.raises(ValueError, match="aggregated"):
        ScenarioGrade("FE-01", Gate.NOT_RUN, activity=(dup, dup))
