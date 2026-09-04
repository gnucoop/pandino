"""Route wiring of embedding Usage attribution for POST /admin/rag-files/upload.

Under test is only what the route owns: which identity - if any - an admin
upload may be attributed to, and the guarantee that the decision is
observational. The persistence lifecycle it feeds is settled elsewhere and
deliberately not re-tested here.

Placed alongside ``test_ingestion_route_usage_attribution.py`` rather than
folded into the /completion.json and /agentchat file, following the
established one-attribution-file-per-route-family organisation.

The property that gives this route its own file is the identity boundary:
the admin session authenticates the *actor*, but the accounting identity is
the one the ratified technical policy provisions. Neither the admin session
username nor ``config.admin.username`` may ever reach a ``users`` lookup,
and absent configuration is the off-switch.

The route declares only
``attribute_usage_to_policy(policy=USAGE_POLICY_ADMIN_RAG_INGESTION)``, and
the behaviour is deliberately asserted end to end through the route,
because it is the route's declared intent - not the boundary's internals -
that these tests protect. The mechanics behind the declaration belong to
usage.attribution, which is why the lookup patch and the diagnostic
logger both name that module.
"""

import io
import logging
from types import SimpleNamespace

from flask import Flask

from routes import admin as admin_route
import usage.attribution as usage_attribution
from utils.logging_config import register_request_context_hooks
from usage.attribution_state import get_usage_attribution

SERVICE = "/admin/rag-files/upload"

TECHNICAL_USERNAME = "__admin_rag_ingestion__"
TECHNICAL_USER_ID = 11

ADMIN_SESSION_USERNAME = "admin-operator"
ADMIN_PASSWORD_HASH = b"$2b$12$adminpasswordhashfixture"
API_KEY = "technical-row-api-key"


def _config(admin_rag_usage_username=TECHNICAL_USERNAME):
    return SimpleNamespace(
        admin_rag_usage_username=admin_rag_usage_username,
        admin=SimpleNamespace(
            username=ADMIN_SESSION_USERNAME,
            password_hash=ADMIN_PASSWORD_HASH,
        ),
        models=SimpleNamespace(
            asr_provider=None,
            asr_base_url=None,
            asr_model=None,
            vision_provider=None,
            vision_model=None,
            completion_embedding_model_provider=None,
            completion_embedding_model=None,
        ),
    )


def _make_app(config=None):
    app = Flask(__name__)
    app.secret_key = "test-secret"
    app.config["MAUI_CONFIG"] = _config() if config is None else config
    register_request_context_hooks(app)
    app.register_blueprint(admin_route.admin_bp)
    return app


def _patch(monkeypatch, captured, user=None, lookup=None):
    """Patch every route boundary to an offline default.

    ``process_rag_file`` records the attribution visible at the moment it
    runs, which is what makes the ordering assertion possible without
    relying on source line numbers.
    """
    if lookup is None:

        def lookup(username):
            captured.setdefault("lookups", []).append(username)
            return user

    # The admin upload does not resolve the technical identity itself:
    # usage.attribution is the sole owner of this callable, so it is
    # the only honest patch point. Patching the route module instead would
    # leave the real lookup live and prove nothing.
    monkeypatch.setattr(usage_attribution, "get_user_by_username", lookup)

    def fake_process_rag_file(*args, **kwargs):
        captured.setdefault("order", []).append("process_rag_file")
        captured["process_called"] = True
        captured["attribution_at_process"] = get_usage_attribution()
        return SimpleNamespace(chunk_count=3)

    monkeypatch.setattr(admin_route, "process_rag_file", fake_process_rag_file)


def _logged_in_client(app):
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["admin_logged_in"] = True
        sess["admin_username"] = ADMIN_SESSION_USERNAME
    return client


def _post(app, **form):
    data = {
        "namespace": "ns",
        "language": "en",
        "file": (io.BytesIO(b"payload-bytes"), "doc.pdf"),
    }
    data.update(form)
    return _logged_in_client(app).post(
        "/admin/rag-files/upload", data=data, content_type="multipart/form-data"
    )


def _diagnostics(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if "event=embedding_usage_attribution_unavailable" in r.getMessage()
    ]


# --------------------------------------------------------------------------
# attribution success
# --------------------------------------------------------------------------


def test_configured_identity_binds_before_any_embedding_work(monkeypatch):
    """The binding invariant: attribution precedes process_rag_file.

    Asserted on observed runtime order, not on source layout: the
    attribution the fake ``process_rag_file`` can see is the one bound
    before it ran.
    """
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={"id": TECHNICAL_USER_ID, "username": TECHNICAL_USERNAME, "client": None},
    )

    response = _post(_make_app())

    assert response.status_code == 302
    assert captured["process_called"] is True

    attribution = captured["attribution_at_process"]
    assert attribution is not None
    assert attribution.user_id == TECHNICAL_USER_ID
    assert attribution.service == SERVICE
    assert attribution.source is None


def test_lookup_uses_only_the_configured_accounting_username(monkeypatch):
    """Neither admin credential ever reaches the users lookup."""
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={"id": TECHNICAL_USER_ID, "username": TECHNICAL_USERNAME, "client": None},
    )

    _post(_make_app())

    assert captured["lookups"] == [TECHNICAL_USERNAME]
    assert ADMIN_SESSION_USERNAME not in captured["lookups"]


def test_source_stays_none_even_when_the_resolved_row_carries_a_client(monkeypatch):
    """Ratified policy is source=None, not the row's client column."""
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={
            "id": TECHNICAL_USER_ID,
            "username": TECHNICAL_USERNAME,
            "client": "coopi",
        },
    )

    _post(_make_app())

    assert captured["attribution_at_process"].source is None


# --------------------------------------------------------------------------
# off-switch and failure modes - all non-blocking
# --------------------------------------------------------------------------


def test_unconfigured_identity_binds_nothing_and_keeps_upload_working(
    monkeypatch, caplog
):
    captured = {}
    _patch(monkeypatch, captured, user={"id": TECHNICAL_USER_ID})
    app = _make_app(_config(admin_rag_usage_username=None))

    with caplog.at_level(logging.WARNING, logger="usage.attribution"):
        response = _post(app)

    assert captured["attribution_at_process"] is None
    assert captured["process_called"] is True
    assert response.status_code == 302
    # The off-switch performs no database lookup at all.
    assert "lookups" not in captured

    diagnostics = _diagnostics(caplog)
    assert len(diagnostics) == 1
    assert "reason=not_configured" in diagnostics[0]
    assert f"service={SERVICE}" in diagnostics[0]


def test_user_not_found_binds_nothing_and_upload_continues(monkeypatch, caplog):
    captured = {}
    _patch(monkeypatch, captured, user=None)

    with caplog.at_level(logging.WARNING, logger="usage.attribution"):
        response = _post(_make_app())

    assert captured["attribution_at_process"] is None
    assert captured["process_called"] is True
    assert response.status_code == 302

    diagnostics = _diagnostics(caplog)
    assert len(diagnostics) == 1
    assert "reason=not_found" in diagnostics[0]


def test_invalid_user_id_binds_nothing_and_upload_continues(monkeypatch, caplog):
    captured = {}
    _patch(monkeypatch, captured, user={"username": TECHNICAL_USERNAME, "id": None})

    with caplog.at_level(logging.WARNING, logger="usage.attribution"):
        response = _post(_make_app())

    assert captured["attribution_at_process"] is None
    assert captured["process_called"] is True
    assert response.status_code == 302

    diagnostics = _diagnostics(caplog)
    assert len(diagnostics) == 1
    assert "reason=invalid_user_id" in diagnostics[0]


def test_lookup_failure_binds_nothing_and_does_not_block_the_upload(
    monkeypatch, caplog
):
    captured = {}

    def failing_lookup(username):
        captured.setdefault("lookups", []).append(username)
        raise RuntimeError("connection refused")

    _patch(monkeypatch, captured, lookup=failing_lookup)

    with caplog.at_level(logging.WARNING, logger="usage.attribution"):
        response = _post(_make_app())

    assert captured["attribution_at_process"] is None
    assert captured["process_called"] is True
    assert response.status_code == 302

    diagnostics = _diagnostics(caplog)
    assert len(diagnostics) == 1
    assert "reason=lookup_failed" in diagnostics[0]
    assert "error_type=RuntimeError" in diagnostics[0]
    assert "connection refused" not in diagnostics[0]


# --------------------------------------------------------------------------
# validation path - attribution is deliberately bound after validation
# --------------------------------------------------------------------------


def test_missing_namespace_keeps_legacy_behaviour_and_never_attributes(monkeypatch):
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={"id": TECHNICAL_USER_ID, "username": TECHNICAL_USERNAME},
    )

    response = _post(_make_app(), namespace="")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/admin/rag-files")
    assert captured.get("process_called") is None
    # No lookup for a request that provably cannot embed anything.
    assert "lookups" not in captured


def test_missing_file_keeps_legacy_behaviour_and_never_attributes(monkeypatch):
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={"id": TECHNICAL_USER_ID, "username": TECHNICAL_USERNAME},
    )

    app = _make_app()
    response = _logged_in_client(app).post(
        "/admin/rag-files/upload",
        data={"namespace": "ns"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 302
    assert captured.get("process_called") is None
    assert "lookups" not in captured


def test_unauthenticated_upload_never_attributes(monkeypatch):
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={"id": TECHNICAL_USER_ID, "username": TECHNICAL_USERNAME},
    )

    app = _make_app()
    response = app.test_client().post(
        "/admin/rag-files/upload",
        data={"namespace": "ns", "file": (io.BytesIO(b"x"), "doc.pdf")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 302
    assert captured.get("process_called") is None
    assert "lookups" not in captured


# --------------------------------------------------------------------------
# diagnostic hygiene
# --------------------------------------------------------------------------


def test_diagnostic_carries_no_identity_or_payload(monkeypatch, caplog):
    """Every secret present in the fixture must be absent from the event.

    Each forbidden value below is genuinely configured on this request -
    the technical username is in AppConfig, the admin username and hash
    are on config.admin and in the session, the API key is on the resolved
    row, and the payload bytes are in the uploaded file - so these are
    real exclusions rather than assertions against values that were never
    there.
    """
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={
            "id": "not-an-int",
            "username": TECHNICAL_USERNAME,
            "api_key": API_KEY,
            "client": None,
        },
    )

    with caplog.at_level(logging.WARNING, logger="usage.attribution"):
        _post(_make_app())

    diagnostics = _diagnostics(caplog)
    assert len(diagnostics) == 1
    message = diagnostics[0]

    assert "reason=invalid_user_id" in message
    for forbidden in (
        TECHNICAL_USERNAME,
        ADMIN_SESSION_USERNAME,
        ADMIN_PASSWORD_HASH.decode("utf-8"),
        API_KEY,
        "payload-bytes",
        "doc.pdf",
        "not-an-int",
    ):
        assert forbidden not in message


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------


def test_route_declares_the_technical_policy_and_owns_no_mechanics():
    """The route names an intent; every mechanism stayed at the boundary."""
    import os

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "routes/admin.py",
    )
    with open(path) as handle:
        source = handle.read()

    assert "attribute_usage_to_policy" in source
    assert "USAGE_POLICY_ADMIN_RAG_INGESTION" in source

    # None of the mechanics the boundary owns: the binder, a private
    # ambient helper, the configuration attribute naming the provisioned
    # identity, and the admin session identity as an attribution key.
    assert "bind_usage_attribution" not in source
    assert "_bind_embedding_usage_attribution" not in source
    assert "admin_rag_usage_username" not in source
    assert "attribute_usage_to_user" not in source
