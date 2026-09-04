"""Route wiring of embedding Usage attribution for POST /storeragfile.

Under test is only what the route owns: which identity - if any - a
/storeragfile request may be attributed to, and the guarantee that the
decision is observational. The persistence lifecycle it feeds is settled
elsewhere and deliberately not re-tested here.

The three request shapes are asserted independently, because their trust
properties differ:

* an explicit non-Dino client is externally authenticated, so its
  already-required userEmail is a real identity;
* an explicit ``client="dino"`` is NOT, so its userEmail is not billable
  and no substitute identity is used;
* the actual legacy Dino fallback - no ``client`` field at all - is the
  one and only path permitted to use the configured technical accounting
  identity, and only while that configuration is present.
"""

import io
import logging
from types import SimpleNamespace

import pytest
from flask import Flask

from routes import ingestion as ingestion_route
from utils import usage_attribution
from utils.logging_config import register_request_context_hooks
from utils.usage_attribution_state import get_usage_attribution

TECHNICAL_USERNAME = "__dino_legacy_ingestion__"
TECHNICAL_USER_ID = 4242
REAL_USER_ID = 77
REAL_USER_EMAIL = "user@example.com"
AUTH_TOKEN = "test-token"
API_KEY = "test-key"


def _config(dino_legacy_usage_username=TECHNICAL_USERNAME):
    return SimpleNamespace(
        dino_legacy_usage_username=dino_legacy_usage_username,
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
    app.config["MAUI_CONFIG"] = _config() if config is None else config
    register_request_context_hooks(app)
    app.register_blueprint(ingestion_route.ingestion_bp)
    return app


def _patch(monkeypatch, captured, user=None, lookup=None):
    """Patch every route boundary to an offline default.

    Authentication always succeeds; ``process_rag_file`` records the
    attribution visible at the moment it runs, which is what makes the
    ordering assertion possible.
    """
    if lookup is None:
        def lookup(username):
            captured.setdefault("lookups", []).append(username)
            return user

    # The ambient attribution lookup moved with the migration: /storeragfile
    # no longer resolves an identity itself, so the only owner of this
    # callable is now utils.usage_attribution. Patching the route module
    # instead would leave the real lookup live and prove nothing.
    monkeypatch.setattr(usage_attribution, "get_user_by_username", lookup)
    monkeypatch.setattr(
        ingestion_route, "dino_authenticate", lambda *a, **k: None
    )
    monkeypatch.setattr(
        ingestion_route, "external_authenticate", lambda *a, **k: None
    )

    def fake_process_rag_file(*args, **kwargs):
        captured["process_called"] = True
        captured["attribution_at_process"] = get_usage_attribution()
        return SimpleNamespace(
            file_id="f1",
            file_name="doc.pdf",
            namespace="ns",
            chunk_count=3,
            language="en",
            tracking_saved=True,
        )

    monkeypatch.setattr(
        ingestion_route, "process_rag_file", fake_process_rag_file
    )


def _post(app, **form):
    data = {
        "authToken": AUTH_TOKEN,
        "graphqlUrl": "https://example.invalid/graphql",
        "url": "https://example.invalid/doc.pdf",
        "file": (io.BytesIO(b"payload-bytes"), "doc.pdf"),
    }
    data.update(form)
    return app.test_client().post(
        "/storeragfile", data=data, content_type="multipart/form-data"
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


def test_explicit_non_dino_client_binds_the_real_authenticated_user(monkeypatch):
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={"id": REAL_USER_ID, "username": REAL_USER_EMAIL, "client": "coopi"},
    )

    response = _post(_make_app(), client="coopi", userEmail=REAL_USER_EMAIL)

    assert response.status_code == 200
    assert captured["lookups"] == [REAL_USER_EMAIL]

    attribution = captured["attribution_at_process"]
    assert attribution is not None
    assert attribution.user_id == REAL_USER_ID
    assert attribution.service == "/storeragfile"
    assert attribution.source == "coopi"


def test_legacy_dino_fallback_binds_the_technical_accounting_identity(monkeypatch):
    """No client field at all: the one path allowed the technical identity.

    The route names only USAGE_POLICY_LEGACY_DINO_INGESTION; the
    provisioned username, its config key and the lookup that resolves it
    all live inside utils.usage_attribution, which is why the lookup trace
    below is captured through that module and not through the route.
    """
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={
            "id": TECHNICAL_USER_ID,
            "username": TECHNICAL_USERNAME,
            "client": "dino",
        },
    )

    response = _post(_make_app())

    assert response.status_code == 200
    assert captured["lookups"] == [TECHNICAL_USERNAME]

    attribution = captured["attribution_at_process"]
    assert attribution is not None
    assert attribution.user_id == TECHNICAL_USER_ID
    assert attribution.service == "/storeragfile"
    assert attribution.source == "dino"


@pytest.mark.parametrize("client_value", ["", None])
def test_empty_and_absent_client_both_take_the_legacy_fallback(
    monkeypatch, client_value
):
    """The captured flag must match the existing ``if not client`` exactly."""
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={"id": TECHNICAL_USER_ID, "username": TECHNICAL_USERNAME, "client": "dino"},
    )

    form = {} if client_value is None else {"client": client_value}
    response = _post(_make_app(), **form)

    assert response.status_code == 200
    assert captured["lookups"] == [TECHNICAL_USERNAME]
    assert captured["attribution_at_process"].user_id == TECHNICAL_USER_ID


def test_attribution_is_bound_before_process_rag_file_runs(monkeypatch):
    """The binding invariant: every contribution is produced inside the call."""
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={"id": REAL_USER_ID, "username": REAL_USER_EMAIL, "client": "coopi"},
    )

    response = _post(_make_app(), client="coopi", userEmail=REAL_USER_EMAIL)

    assert response.status_code == 200
    assert captured["process_called"] is True
    assert captured["attribution_at_process"] is not None


# --------------------------------------------------------------------------
# the trust boundary
# --------------------------------------------------------------------------


def test_explicit_dino_binds_nothing_and_never_trusts_its_user_email(
    monkeypatch, caplog
):
    """Explicit Dino is not the legacy fallback and has no usable identity.

    The route declares ``declare_usage_unattributed()`` here, whose whole
    observable contract is negative: nothing bound, no technical-identity
    lookup, no ambient lookup of the unverified userEmail, and - the
    property that separates a deliberate absence from every degradation
    path - no diagnostic at all. Capture is left unfiltered so a record
    from any logger would surface.
    """
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={"id": REAL_USER_ID, "username": REAL_USER_EMAIL, "client": "coopi"},
    )

    with caplog.at_level(logging.WARNING):
        response = _post(_make_app(), client="dino", userEmail=REAL_USER_EMAIL)

    assert response.status_code == 200
    assert captured["process_called"] is True
    assert "lookups" not in captured
    assert captured["attribution_at_process"] is None
    # Expected steady state, not a failure: stays quiet.
    assert _diagnostics(caplog) == []


# --------------------------------------------------------------------------
# failure isolation
# --------------------------------------------------------------------------


def test_legacy_fallback_without_configuration_is_the_off_switch(monkeypatch, caplog):
    captured = {}
    _patch(monkeypatch, captured, user={"id": TECHNICAL_USER_ID, "client": "dino"})
    app = _make_app(_config(dino_legacy_usage_username=None))

    with caplog.at_level(logging.WARNING, logger="utils.usage_attribution"):
        response = _post(app)

    assert response.status_code == 200
    assert "lookups" not in captured
    assert captured["attribution_at_process"] is None

    messages = _diagnostics(caplog)
    assert len(messages) == 1
    assert "reason=not_configured" in messages[0]
    assert "service=/storeragfile" in messages[0]


def test_user_not_found_binds_nothing_and_leaves_ingestion_unchanged(
    monkeypatch, caplog
):
    captured = {}
    _patch(monkeypatch, captured, user=None)

    with caplog.at_level(logging.WARNING, logger="utils.usage_attribution"):
        response = _post(_make_app(), client="coopi", userEmail=REAL_USER_EMAIL)

    assert response.status_code == 200
    assert captured["process_called"] is True
    assert captured["attribution_at_process"] is None

    messages = _diagnostics(caplog)
    assert len(messages) == 1
    assert "reason=not_found" in messages[0]


def test_invalid_user_id_binds_nothing(monkeypatch, caplog):
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={"id": "not-an-int", "username": REAL_USER_EMAIL, "client": "coopi"},
    )

    with caplog.at_level(logging.WARNING, logger="utils.usage_attribution"):
        response = _post(_make_app(), client="coopi", userEmail=REAL_USER_EMAIL)

    assert response.status_code == 200
    assert captured["attribution_at_process"] is None

    messages = _diagnostics(caplog)
    assert len(messages) == 1
    assert "reason=invalid_user_id" in messages[0]


def test_lookup_failure_binds_nothing_and_never_fails_the_request(
    monkeypatch, caplog
):
    captured = {}

    def failing_lookup(username):
        captured.setdefault("lookups", []).append(username)
        raise RuntimeError("database is down")

    _patch(monkeypatch, captured, lookup=failing_lookup)

    with caplog.at_level(logging.WARNING, logger="utils.usage_attribution"):
        response = _post(_make_app(), client="coopi", userEmail=REAL_USER_EMAIL)

    assert response.status_code == 200
    assert captured["process_called"] is True
    assert captured["attribution_at_process"] is None

    messages = _diagnostics(caplog)
    assert len(messages) == 1
    assert "reason=lookup_failed" in messages[0]
    assert "error_type=RuntimeError" in messages[0]


def test_attribution_diagnostic_carries_no_identity_or_payload(monkeypatch, caplog):
    captured = {}

    def failing_lookup(username):
        raise RuntimeError("database is down")

    _patch(monkeypatch, captured, lookup=failing_lookup)

    with caplog.at_level(logging.WARNING, logger="utils.usage_attribution"):
        _post(_make_app(), client="coopi", userEmail=REAL_USER_EMAIL)

    message = _diagnostics(caplog)[0]
    for forbidden in (
        REAL_USER_EMAIL,
        TECHNICAL_USERNAME,
        AUTH_TOKEN,
        API_KEY,
        "doc.pdf",
        "payload-bytes",
    ):
        assert forbidden not in message


def test_technical_username_never_appears_in_the_fallback_diagnostic(
    monkeypatch, caplog
):
    captured = {}
    _patch(monkeypatch, captured, user=None)

    with caplog.at_level(logging.WARNING, logger="utils.usage_attribution"):
        response = _post(_make_app())

    assert response.status_code == 200
    message = _diagnostics(caplog)[0]
    assert "reason=not_found" in message
    assert TECHNICAL_USERNAME not in message


# --------------------------------------------------------------------------
# the HTTP contract is untouched
# --------------------------------------------------------------------------


def test_success_response_shape_is_unchanged(monkeypatch):
    captured = {}
    _patch(
        monkeypatch,
        captured,
        user={"id": REAL_USER_ID, "username": REAL_USER_EMAIL, "client": "coopi"},
    )

    response = _post(_make_app(), client="coopi", userEmail=REAL_USER_EMAIL)

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "file_id": "f1",
        "file_name": "doc.pdf",
        "namespace": "ns",
        "chunk_count": 3,
        "language": "en",
        "tracking_saved": True,
    }


def test_missing_auth_token_still_400s_before_any_attribution(monkeypatch):
    captured = {}
    _patch(monkeypatch, captured, user={"id": REAL_USER_ID, "client": "dino"})

    response = _post(_make_app(), authToken="")

    assert response.status_code == 400
    assert "lookups" not in captured


def test_missing_user_email_still_400s_for_a_non_dino_client(monkeypatch):
    captured = {}
    _patch(monkeypatch, captured, user={"id": REAL_USER_ID, "client": "coopi"})

    response = _post(_make_app(), client="coopi")

    assert response.status_code == 400
    assert "lookups" not in captured


def test_missing_graphql_url_still_400s_for_dino(monkeypatch):
    captured = {}
    _patch(monkeypatch, captured, user={"id": TECHNICAL_USER_ID, "client": "dino"})

    response = _post(_make_app(), graphqlUrl="")

    assert response.status_code == 400
    assert "lookups" not in captured


def test_authentication_failure_still_403s_and_binds_nothing(monkeypatch):
    captured = {}
    _patch(monkeypatch, captured, user={"id": TECHNICAL_USER_ID, "client": "dino"})
    monkeypatch.setattr(
        ingestion_route, "dino_authenticate", lambda *a, **k: "not authorised"
    )

    response = _post(_make_app())

    assert response.status_code == 403
    assert "lookups" not in captured
    assert "process_called" not in captured


def test_processing_failure_status_codes_are_unchanged(monkeypatch):
    captured = {}
    _patch(monkeypatch, captured, user={"id": TECHNICAL_USER_ID, "client": "dino"})

    def raising(*args, **kwargs):
        raise ValueError("unsupported file type")

    monkeypatch.setattr(ingestion_route, "process_rag_file", raising)

    response = _post(_make_app())

    assert response.status_code == 400
    assert response.get_data(as_text=True) == "unsupported file type"


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------


def test_route_declares_attribution_but_never_persists_usage_itself():
    """Persistence stays lifecycle-owned; the route only declares intent."""
    import os

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "routes/ingestion.py",
    )
    with open(path) as handle:
        source = handle.read()

    assert "bind_usage_attribution" not in source
    assert "_bind_embedding_usage_attribution" not in source
    assert "attribute_usage_to_policy" in source
    assert "log_resolved_cost_usage_batch" not in source
    assert "register_usage_log_id" not in source
