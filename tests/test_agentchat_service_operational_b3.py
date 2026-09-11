"""FOURTH ADOPTER SLICE B3 — /agentchat service-owned facts.

Two facts, both owned by services/agentchat_service.py::run_agentchat:

    agentchat_agent_failed       (ERROR,   error_type)
    agentchat_audit_log_failed   (WARNING, error_type)

E4 exists for exactly one reason (§12.3): `:142`'s `except Exception` is the
ONLY point in the process where the REAL exception class still exists. The
next statement wraps it into a `RuntimeError` message string, and the route
(B2) is structurally incapable of recovering the class from that — every
service failure reaches it as `RuntimeError`. E4 therefore carries
`error_type` and nothing else: `start_time` is unbound on most paths reaching
this handler, and provider/model availability holds only under a production-
shape assumption, so referencing them here could raise a `NameError` from
inside the exception handler and destroy the very fact being recorded.

E5 reuses the EXISTING runtime token `agentchat_audit_log_failed` (C17): the
semantic contract is identical, so the legacy line at `:132` is REPLACED, not
supplemented. It records the LOSS of the audit channel, never its content, and
it must remain fail-open — `run_agentchat` still returns its already-built
result.

B3 is diagnostic only. The RuntimeError wrapping contract, the explicit
`except RuntimeError` re-raise arm, fail-open audit behaviour, `log_runresult`
and `agent_runs` are all unchanged.
"""

import json
import logging
from types import SimpleNamespace

import pytest

from services import agentchat_service
from utils.logging_config import LOG_FORMAT, ContextDefaultsFilter, UtcIsoFormatter
from utils.operational_persistence import (
    OperationalPersistenceHandler,
    snapshot_from_record,
)

_E4 = "agentchat_agent_failed"
_E5 = "agentchat_audit_log_failed"

QUESTION = "SUPERSECRETQUESTION"
USERNAME = "distinctive-user@example.com"

_SERVICE_LOGGER = "services.agentchat_service"


class _StubAgent:
    """Stand-in for smolagents.CodeAgent: no LLM, no tool calls."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        return SimpleNamespace(
            steps=[],
            timing=None,
            token_usage=None,
            state=None,
            output={"answer": "real answer", "follow_ups": []},
        )


def _stub_config():
    models = SimpleNamespace(
        completion_model_provider="stub-provider",
        completion_model_agent_chat="stub-model",
        completion_embedding_model_provider="stub-emb-provider",
        completion_embedding_model="stub-emb-model",
    )
    rag = SimpleNamespace(top_k=3, min_sim=0.5)
    return SimpleNamespace(models=models, rag=rag)


def _patch_service_seams(monkeypatch, *, agent=_StubAgent):
    """Only external seams: the LLM client, the retriever, the agent, prompt
    loading and serialization. Every boundary under test stays real."""
    monkeypatch.setattr(
        agentchat_service, "build_litellm_model", lambda **kwargs: object()
    )
    monkeypatch.setattr(agentchat_service, "RetrieverTool", lambda **kwargs: object())
    monkeypatch.setattr(agentchat_service, "CodeAgent", agent)
    monkeypatch.setattr(agentchat_service, "load_prompt", lambda *a, **kw: "template")
    monkeypatch.setattr(agentchat_service, "render_prompt", lambda *a, **kw: "rendered")
    monkeypatch.setattr(
        agentchat_service,
        "serialize_runresult",
        lambda result: {
            "answer": "real answer",
            "follow_ups": [],
            "tool_calls": [],
            "vectors": [],
            "metrics": {"token_usage": {}},
        },
    )
    monkeypatch.setattr(agentchat_service, "log_runresult", lambda *a, **kw: None)


def _run():
    return agentchat_service.run_agentchat(
        chat=[QUESTION],
        namespace="agentchat",
        language="ENG",
        username=USERNAME,
        config=_stub_config(),
    )


def _operational_records(caplog, event):
    return [
        r
        for r in caplog.records
        if getattr(r, "maui_persist", None) is True
        and getattr(r, "maui_event", None) == event
    ]


def _the_record(caplog, event):
    records = _operational_records(caplog, event)
    assert len(records) == 1, (
        f"expected exactly one {event} record, got {len(records)}"
    )
    return records[0]


def _assert_error_type_only(record, *, event, error_type, level):
    """The ratified single-field contract for BOTH B3 events.

    `error_type` is present and correct; provider, model, duration_ms, details
    and message are all ABSENT (R25/R26), on the LogRecord and in the persisted
    snapshot alike. request_id/app_id stay foundation-owned — the call site
    contributes the class name and nothing else.
    """
    assert record.maui_error_type == error_type
    for field in (
        "maui_provider",
        "maui_model",
        "maui_duration_ms",
        "maui_details",
        "maui_message",
    ):
        assert not hasattr(record, field), f"{field} must be absent from {event}"

    snapshot = snapshot_from_record(record)
    assert snapshot is not None
    assert snapshot.event == event
    assert snapshot.level == level
    assert snapshot.error_type == error_type
    assert snapshot.provider is None
    assert snapshot.model is None
    assert snapshot.duration_ms is None
    assert snapshot.message is None
    assert snapshot.details_json in (None, "null") or json.loads(
        snapshot.details_json
    ) in (None, {})
    return snapshot


def _assert_nothing_leaked(record, snapshot, *sentinels):
    surfaces = [
        record.getMessage(),
        str(getattr(record, "maui_details", None)),
        str(snapshot.details_json),
        str(snapshot.message),
        str(snapshot.error_type),
        str(snapshot),
    ]
    for surface in surfaces:
        assert "Traceback" not in surface, (
            f"a traceback reached an Operational surface: {surface!r}"
        )
        for needle in sentinels:
            assert needle not in surface, (
                f"forbidden content {needle!r} reached an Operational "
                f"surface: {surface!r}"
            )


# ---------------------------------------------------------------------------
# E4 — the real service failure class
# ---------------------------------------------------------------------------


def test_agent_run_failure_persists_the_real_class(monkeypatch, caplog):
    """§12.3 / test 10 — `agent.run` raises a non-RuntimeError.

    Exactly one E4 carrying the GENUINE class, the RuntimeError wrap still
    propagates unchanged, and no provider/model/duration_ms/details/message
    reaches the persisted row.
    """
    sentinel = "UPSTREAM-BODY-secret-4242"

    class _BoomAgent(_StubAgent):
        def run(self, *args, **kwargs):
            raise ValueError(f"bad value from {QUESTION} :: {sentinel}")

    _patch_service_seams(monkeypatch, agent=_BoomAgent)

    with caplog.at_level(logging.INFO, logger=_SERVICE_LOGGER):
        with pytest.raises(RuntimeError) as excinfo:
            _run()

    # The wrapping contract is untouched: same RuntimeError, same chaining.
    assert str(excinfo.value).startswith("agentchat_service failed:")
    assert isinstance(excinfo.value.__cause__, ValueError)

    record = _the_record(caplog, _E4)
    assert record.levelno == logging.ERROR
    snapshot = _assert_error_type_only(
        record, event=_E4, error_type="ValueError", level="ERROR"
    )
    _assert_nothing_leaked(record, snapshot, sentinel, QUESTION, USERNAME)

    # logger.exception: runtime depth at a previously UNLOGGED boundary.
    assert record.exc_info is not None
    assert _operational_records(caplog, _E5) == []


def test_failure_before_start_time_is_bound_still_emits_e4(monkeypatch, caplog):
    """§12.3 — the NameError regression that removed provider/model/duration.

    `build_litellm_model` (`:67`) is the first operation that can raise a
    non-RuntimeError, and it runs long BEFORE `start_time` is bound at `:101`.
    If E4 ever referenced start_time/duration_ms — or any local not bound on
    every reachable path — this handler would raise `NameError` and REPLACE the
    diagnosable failure with an opaque one. E4 must emit the real class here
    exactly as it does on the agent.run path.
    """

    class _ModelBuildError(Exception):
        pass

    _patch_service_seams(monkeypatch)

    def _explode(**kwargs):
        raise _ModelBuildError("no credentials")

    monkeypatch.setattr(agentchat_service, "build_litellm_model", _explode)

    with caplog.at_level(logging.INFO, logger=_SERVICE_LOGGER):
        with pytest.raises(RuntimeError) as excinfo:
            _run()

    # The original exception survived: no NameError replaced it.
    assert isinstance(excinfo.value.__cause__, _ModelBuildError)
    assert not isinstance(excinfo.value.__cause__, NameError)

    record = _the_record(caplog, _E4)
    assert record.levelno == logging.ERROR
    _assert_error_type_only(
        record, event=_E4, error_type="_ModelBuildError", level="ERROR"
    )


def test_real_logger_exception_splits_traceback_from_snapshot(monkeypatch):
    """§13.3.1 / I8 — B3's boundary-faithful test for E4.

    One REAL `logger.exception(message, extra=extra)` call, raised from the
    service's real `except Exception`, must give stderr the full traceback
    while the Operational snapshot keeps `error_type` only. Both halves are
    asserted together, because either alone would pass while the split
    silently broke.
    """
    sentinel = "AGENT-INTERNALS-secret-9191"

    class AgentBoom(Exception):
        pass

    class _BoomAgent(_StubAgent):
        def run(self, *args, **kwargs):
            raise AgentBoom(f"agent said {sentinel}")

    _patch_service_seams(monkeypatch, agent=_BoomAgent)

    sink = []
    handler = OperationalPersistenceHandler(sink.append)
    service_logger = logging.getLogger(_SERVICE_LOGGER)

    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    capture = _Capture()
    capture.addFilter(ContextDefaultsFilter())

    previous_level = service_logger.level
    service_logger.addHandler(handler)
    service_logger.addHandler(capture)
    service_logger.setLevel(logging.INFO)
    try:
        with pytest.raises(RuntimeError):
            _run()
    finally:
        service_logger.removeHandler(handler)
        service_logger.removeHandler(capture)
        service_logger.setLevel(previous_level)

    records = [r for r in captured if getattr(r, "maui_event", None) == _E4]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.ERROR

    # --- stderr half: the real formatter keeps the traceback ----------------
    assert record.exc_info is not None
    formatted = UtcIsoFormatter(LOG_FORMAT).format(record)
    assert "Traceback" in formatted
    assert "AgentBoom" in formatted
    assert sentinel in formatted
    assert record.exc_text and sentinel in record.exc_text

    # --- Operational half: same record, bounded snapshot --------------------
    snapshots = [s for s in sink if s.event == _E4]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.level == "ERROR"
    assert snapshot.error_type == "AgentBoom"
    assert snapshot.provider is None
    assert snapshot.model is None
    assert snapshot.duration_ms is None
    assert snapshot.message is None
    assert sentinel not in str(snapshot)
    assert "Traceback" not in str(snapshot)

    # Taken again AFTER stderr formatting cached exc_text on the shared
    # record, the snapshot is still clean.
    post_format = snapshot_from_record(record)
    assert post_format.error_type == "AgentBoom"
    assert sentinel not in str(post_format)
    assert "Traceback" not in str(post_format)


def test_model_configuration_guard_emits_no_e4(monkeypatch, caplog):
    """§12.3 / G6 / test 11 — an INTENTIONAL non-emission. DO NOT "fix" this.

    The model-configuration guard raises `RuntimeError` directly, so the
    explicit `except RuntimeError: raise` arm re-raises it BEFORE the generic
    handler that owns E4 is ever entered. There is nothing for E4 to preserve:
    the class genuinely IS `RuntimeError` at that boundary.

    The absence of an E4 row here is DESIGNED SILENCE, not missing coverage.
    The terminal outcome of such a request is still represented — by the
    route-level B2 fact `agentchat_uncontrolled_failure{reason=service_error}`.
    """
    _patch_service_seams(monkeypatch)
    config = _stub_config()
    config.models.completion_model_agent_chat = ""

    with caplog.at_level(logging.INFO, logger=_SERVICE_LOGGER):
        with pytest.raises(RuntimeError) as excinfo:
            agentchat_service.run_agentchat(
                chat=[QUESTION],
                namespace="agentchat",
                language="ENG",
                username=USERNAME,
                config=config,
            )

    # Propagation is unchanged: the guard's own message, not a wrapped one.
    assert str(excinfo.value) == "COMPLETION_MODEL_AGENT_CHAT is not configured."
    assert excinfo.value.__cause__ is None

    assert _operational_records(caplog, _E4) == [], (
        "the explicit RuntimeError arm must emit NO agentchat_agent_failed — "
        "see G6; this silence is designed"
    )
    assert _operational_records(caplog, _E5) == []


# ---------------------------------------------------------------------------
# E5 — loss of the agent_runs audit write
# ---------------------------------------------------------------------------


def test_audit_failure_persists_error_type_and_stays_fail_open(monkeypatch, caplog):
    """§12.4 / test 12 — the contained audit failure.

    Exactly one E5 at WARNING carrying the real audit exception class, and —
    load-bearing — `run_agentchat` STILL RETURNS its already-built result. A
    malformed builder call in this handler would turn a contained audit failure
    into a request failure, which is precisely what this asserts against.
    """
    _patch_service_seams(monkeypatch)

    answer_sentinel = "AUDIT-PAYLOAD-secret-7373"

    def _raise_on_log(*args, **kwargs):
        raise OSError(f"audit unavailable while writing {answer_sentinel}")

    monkeypatch.setattr(agentchat_service, "log_runresult", _raise_on_log)

    with caplog.at_level(logging.INFO, logger=_SERVICE_LOGGER):
        result = _run()

    # FAIL-OPEN: the response is unchanged and complete.
    assert result["payload"]["answer"] == "real answer"
    assert result["model"] == "stub-model"
    assert result["provider"] == "stub-provider"

    record = _the_record(caplog, _E5)
    assert record.levelno == logging.WARNING
    snapshot = _assert_error_type_only(
        record, event=_E5, error_type="OSError", level="WARNING"
    )
    _assert_nothing_leaked(record, snapshot, answer_sentinel, QUESTION, USERNAME)

    # E5 replaces the legacy line; it does not supplement it.
    assert _operational_records(caplog, _E4) == []
    for captured in caplog.records:
        assert "error=audit unavailable" not in captured.getMessage(), (
            "the replaced legacy runtime line is still being emitted"
        )


def test_real_logger_warning_exc_info_splits_traceback_from_snapshot(monkeypatch):
    """§12.4 / I8 — B3's boundary-faithful test for E5's `exc_info=True`.

    The audit exception is swallowed, so WITHOUT `exc_info=True` its context
    would be lost entirely: nothing else logs it and nothing re-raises it. One
    real `logger.warning(message, extra=extra, exc_info=True)` must therefore
    put the traceback on stderr while the persisted snapshot carries neither
    the traceback nor the raw exception text.
    """
    sentinel = "AUDIT-INTERNALS-secret-5151"

    class AuditBoom(Exception):
        pass

    _patch_service_seams(monkeypatch)

    def _raise_on_log(*args, **kwargs):
        raise AuditBoom(f"audit store said {sentinel}")

    monkeypatch.setattr(agentchat_service, "log_runresult", _raise_on_log)

    sink = []
    handler = OperationalPersistenceHandler(sink.append)
    service_logger = logging.getLogger(_SERVICE_LOGGER)

    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    capture = _Capture()
    capture.addFilter(ContextDefaultsFilter())

    previous_level = service_logger.level
    service_logger.addHandler(handler)
    service_logger.addHandler(capture)
    service_logger.setLevel(logging.INFO)
    try:
        result = _run()
    finally:
        service_logger.removeHandler(handler)
        service_logger.removeHandler(capture)
        service_logger.setLevel(previous_level)

    # Still fail-open through the REAL logging boundary.
    assert result["payload"]["answer"] == "real answer"

    records = [r for r in captured if getattr(r, "maui_event", None) == _E5]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING

    # --- stderr half ---------------------------------------------------------
    assert record.exc_info is not None, "exc_info=True is load-bearing for E5"
    formatted = UtcIsoFormatter(LOG_FORMAT).format(record)
    assert "Traceback" in formatted
    assert "AuditBoom" in formatted
    assert sentinel in formatted

    # --- Operational half ----------------------------------------------------
    snapshots = [s for s in sink if s.event == _E5]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.level == "WARNING"
    assert snapshot.error_type == "AuditBoom"
    assert snapshot.message is None
    assert snapshot.provider is None
    assert snapshot.model is None
    assert snapshot.duration_ms is None
    assert sentinel not in str(snapshot)
    assert "Traceback" not in str(snapshot)

    post_format = snapshot_from_record(record)
    assert post_format.error_type == "AuditBoom"
    assert sentinel not in str(post_format)
    assert "Traceback" not in str(post_format)


# ---------------------------------------------------------------------------
# Route-level pairing — B3 must not disturb B2
# ---------------------------------------------------------------------------


def _make_app():
    from flask import Flask

    from routes import rag as rag_route
    from utils.logging_config import register_request_context_hooks

    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = _stub_config()
    app.config["MAUI_CONFIG"].rag.default_namespace = "default-ns"
    app.config["MAUI_CONFIG"].completion_token_cost = 1
    register_request_context_hooks(app)
    app.register_blueprint(rag_route.rag_bp)
    return app


@pytest.mark.parametrize(
    "break_agent, expect_e4",
    [
        # agent.run raises a genuine non-RuntimeError → E4 + E3{service_error}
        (True, True),
        # the model-configuration guard fires → E3{service_error} ONLY (G6)
        (False, False),
    ],
)
def test_route_still_records_service_error_alongside_b3(
    monkeypatch, caplog, break_agent, expect_e4
):
    """§12.3 / G6 — E4 and B2's terminal fact are complementary, not exclusive.

    E4 answers *with what class did the service fail*; B2's
    `agentchat_uncontrolled_failure{reason=service_error}` remains the terminal
    REQUEST outcome and is unchanged by this slice. On the guard path E4 is
    intentionally silent and B2's fact alone represents the failure.
    """
    from routes import rag as rag_route

    class _BoomAgent(_StubAgent):
        def run(self, *args, **kwargs):
            raise ValueError("agent exploded")

    _patch_service_seams(monkeypatch, agent=_BoomAgent if break_agent else _StubAgent)
    monkeypatch.setattr(rag_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(rag_route, "attribute_usage_to_user", lambda **k: None)
    monkeypatch.setattr(
        rag_route.database_pg, "get_user_tokens", lambda username: 10
    )

    app = _make_app()
    if not break_agent:
        app.config["MAUI_CONFIG"].models.completion_model_agent_chat = ""

    with caplog.at_level(logging.INFO):
        response = app.test_client().post(
            "/agentchat",
            json={"chat": [QUESTION], "username": USERNAME},
            headers={"X-API-KEY": "k"},
        )

    assert response.status_code == 500

    terminal = _operational_records(caplog, "agentchat_uncontrolled_failure")
    assert len(terminal) == 1
    assert terminal[0].maui_details == {"reason": "service_error"}
    assert not hasattr(terminal[0], "maui_error_type")

    e4 = _operational_records(caplog, _E4)
    if expect_e4:
        assert len(e4) == 1
        assert e4[0].maui_error_type == "ValueError"
    else:
        assert e4 == [], "G6: the RuntimeError guard path emits no E4"
