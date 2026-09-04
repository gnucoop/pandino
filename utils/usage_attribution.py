# utils/usage_attribution.py
"""Public adoption boundary for Usage attribution.

This is the request-scoped surface a Maui flow uses to say *whose*
provider consumption this request produces. It is a sibling of
``utils.usage_recording``, which says *what* was consumed, and it is
layered above the storage-only ``utils.usage_attribution_state``, exactly
as recording is layered above ``utils.usage_request_state``.

There are three attribution intents, and a caller declares the one that is
semantically true for its flow::

    attribute_usage_to_user(username=...)

    attribute_usage_to_policy(policy=USAGE_POLICY_LEGACY_DINO_INGESTION)
    attribute_usage_to_policy(policy=USAGE_POLICY_ADMIN_RAG_INGESTION)

    declare_usage_unattributed()

A caller declares intent and nothing more. Which point in the flow the
intent becomes valid stays the caller's decision - there is no automatic
hook - but every mechanism behind the declaration belongs here.

Identity is the caller's warrant, not this module's finding
----------------------------------------------------------
:func:`attribute_usage_to_user` records an accounting assignment for an
identity **the caller has already authenticated**. This module performs no
authentication of its own: it does not verify a password, a token, a
session or an external principal, and resolving a ``users`` row is a
lookup, never a proof. Passing an unverified value therefore attributes
consumption to an identity nobody checked, which is why a flow holding an
unverified principal must declare
:func:`declare_usage_unattributed` instead of attributing it.

What this boundary owns, so no adopter has to
----------------------------------------------
* **Identity resolution.** The ``users``-row lookup, the shape of the row
  it returns, and validation of the persistent ``users.id``. That id is
  used here and never returned, so a caller cannot hold one.
* **Source policy.** Whether the bound ``source`` derives from the
  resolved row's ``client`` column or is explicitly ``None`` is a property
  of the declared intent, decided here.
* **Technical identity provisioning.** Each technical policy maps to the
  configuration attribute naming its provisioned accounting identity.
  Callers name the policy; the provisioned username and its configuration
  key appear in no call site. Absent configuration is the off-switch.
* **Service derivation.** ``service`` is the request's registered route
  identity, read from the active Flask request. There is deliberately no
  keyword for it, so an attempt to supply one fails as a ``TypeError`` at
  the call site. The persistence layer below stays request-blind and
  continues to receive a plain string.
* **Request-scoped binding.** ``UsageAttribution`` and
  ``bind_usage_attribution`` are subsystem mechanics; a route should not
  need to know either name.
* **Diagnostics and redaction.** One safe event, with a fixed and narrow
  field set.

Failure model
-------------
Two kinds of failure, deliberately separated, with validation running
before fail-open containment - the discipline of
``utils.usage_recording`` and ``utils.operational_event``.

*Programmer-contract misuse* raises. A blank, whitespace-only or non-``str``
username raises ``ValueError``; so does a policy value outside the closed
approved vocabulary, because a policy is authored as a module constant at
the call site and a silent unknown would leave that vocabulary open. A
misspelled or positional argument raises ``TypeError``, since the
signatures are keyword-only and absorb no ``**kwargs``.

*Runtime degradation* never raises and never retries. Attribution is
observational: no request context, an unprovisioned technical identity, a
missing ``users`` row, a non-``int`` persistent id, a failing lookup and a
failing bind all produce the same outcome - nothing bound, one safe
diagnostic, and application flow entirely unaffected. There is no retry,
consistent with the rest of the Usage subsystem.

Return contract
---------------
All three operations return ``None``. Attribution is observation, and no
application or business decision may branch on whether it succeeded;
exposing success would invite exactly that coupling, which fail-open
exists to prevent.

**The asymmetry with Explicit Usage Recording is intentional.**
``record_token_consumption`` and ``record_resolved_consumption`` return
``bool`` because a Usage row either was persisted or was not, and some
routes legitimately surface that. Attribution is different in kind: it is
metadata bound mid-request for a row that may not yet exist, and no HTTP
response depends on it.
"""

import logging

from flask import current_app, has_request_context, request

from infrastructure.database_pg import get_user_by_username
from utils.logging_config import get_request_id
from utils.usage_attribution_state import bind_usage_attribution

__all__ = [
    "attribute_usage_to_user",
    "attribute_usage_to_policy",
    "declare_usage_unattributed",
    "USAGE_POLICY_LEGACY_DINO_INGESTION",
    "USAGE_POLICY_ADMIN_RAG_INGESTION",
]

logger = logging.getLogger(__name__)

#: Ingestion that arrived through the legacy Dino fallback - no ``client``
#: field supplied at all - and carries no verifiable end-user identity.
USAGE_POLICY_LEGACY_DINO_INGESTION = "legacy_dino_ingestion"

#: Admin-initiated RAG ingestion, which has an operator but no end user.
USAGE_POLICY_ADMIN_RAG_INGESTION = "admin_rag_ingestion"

#: The closed technical-policy vocabulary: policy -> (configuration
#: attribute naming the provisioned identity, whether ``source`` derives
#: from the resolved row's ``client``).
#:
#: The two source rules differ deliberately and are not one global
#: technical-user rule: legacy Dino ingestion reports the client its
#: provisioned row carries, while admin RAG ingestion has no client concept
#: to report and binds ``None`` whatever the resolved row happens to hold.
#:
#: A plain table, read only here. There is no dynamic registration: a third
#: policy is a code change and its own ratification, and that friction is
#: intended.
_TECHNICAL_POLICIES = {
    USAGE_POLICY_LEGACY_DINO_INGESTION: ("dino_legacy_usage_username", True),
    USAGE_POLICY_ADMIN_RAG_INGESTION: ("admin_rag_usage_username", False),
}


def _require_non_empty_str(name: str, value) -> str:
    """Require a str carrying something other than whitespace.

    Whitespace is rejected as well as emptiness: a blank identity key
    resolves nothing and would silently fragment accounting rather than
    failing where the defect is.
    """
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty str")
    return value


def _derive_service() -> "str | None":
    """Return the active request's registered route identity.

    The only Flask-aware derivation in the Usage attribution path, and the
    reason it lives at this boundary rather than below it: endpoint
    identity belongs to the request context this module already reads for
    ``request_id``. ``request.url_rule.rule`` is preferred over
    ``request.path`` because it stays the route's identity even if a route
    ever gains a dynamic segment.

    :return: the route rule, or ``None`` when there is no request context
        or the request matched no rule - a runtime degradation, not misuse.
    """
    if not has_request_context():
        return None
    url_rule = request.url_rule
    if url_rule is None:
        return None
    return url_rule.rule


def _report_attribution_unavailable(
    reason: str, service: "str | None", error_type: "str | None" = None
) -> None:
    """Emit the one established, identity-safe attribution diagnostic.

    Carries only ``reason``, ``service``, ``request_id`` and ``error_type``:
    never a real username or email, never the configured technical or admin
    username, never a password or hash, an API key, an auth token, a
    filename or anything from the request payload.
    """
    logger.warning(
        "event=embedding_usage_attribution_unavailable reason=%s "
        "service=%s request_id=%s error_type=%s",
        reason,
        service,
        get_request_id(),
        error_type,
    )


def _resolve_and_bind(username: str, service: str, derive_source: bool) -> None:
    """Resolve ``username`` to a persistent id and bind the attribution.

    The one shared tail of every attributing intent: look the row up,
    validate the persistent id, resolve ``source`` under the caller's rule,
    bind. Whether the username is a real authenticated identity or a
    provisioned technical one makes no difference here, and neither ever
    reaches a diagnostic.

    Fail-open in full: any degradation binds nothing, emits one safe
    diagnostic and returns.
    """
    reason = None
    error_type = None
    try:
        user = get_user_by_username(username)
        if not user:
            reason = "not_found"
        else:
            user_id = user.get("id")
            if isinstance(user_id, int):
                source = user.get("client") if derive_source else None
                bind_usage_attribution(user_id, service, source)
            else:
                reason = "invalid_user_id"
    except Exception as exc:
        reason = "lookup_failed"
        error_type = type(exc).__name__

    if reason is not None:
        _report_attribution_unavailable(reason, service, error_type)


def attribute_usage_to_user(*, username: str) -> None:
    """Attribute this request's consumption to an authenticated identity.

    Says: *this provider consumption belongs to this identity, which the
    caller has already authenticated.* The username is the lookup key every
    adopter already holds; this module resolves it, and deliberately does
    not verify it - see the module docstring.

    ``source`` is taken from the resolved row's ``client`` column, never
    hardcoded.

    :param username: an already-authenticated identity's username.
    :raises ValueError: if ``username`` is not a non-blank ``str``.
    """
    username = _require_non_empty_str("username", username)

    service = _derive_service()
    if service is None:
        _report_attribution_unavailable("no_request_context", None)
        return

    _resolve_and_bind(username, service, derive_source=True)


def attribute_usage_to_policy(*, policy: str) -> None:
    """Attribute this request's consumption to a technical accounting policy.

    Says: *this consumption belongs to this approved technical accounting
    policy.* The caller names one of the module's policy constants; which
    provisioned identity serves it, and what ``source`` that policy
    reports, are decided here.

    An unprovisioned policy is the intended off-switch: nothing is bound,
    one ``not_configured`` diagnostic is emitted, and no ``users`` lookup is
    performed at all.

    :param policy: one of :data:`USAGE_POLICY_LEGACY_DINO_INGESTION` or
        :data:`USAGE_POLICY_ADMIN_RAG_INGESTION`.
    :raises ValueError: if ``policy`` is not one of those constants. The
        vocabulary is closed, so an unknown value is misuse to surface
        during development, not a runtime condition to absorb.
    """
    if not isinstance(policy, str) or policy not in _TECHNICAL_POLICIES:
        raise ValueError(
            f"unknown policy: {policy!r}; expected one of "
            f"{sorted(_TECHNICAL_POLICIES)}"
        )

    service = _derive_service()
    if service is None:
        _report_attribution_unavailable("no_request_context", None)
        return

    config_attribute, derive_source = _TECHNICAL_POLICIES[policy]
    try:
        technical_username = getattr(
            current_app.config["MAUI_CONFIG"], config_attribute, None
        )
    except Exception as exc:
        _report_attribution_unavailable("lookup_failed", service, type(exc).__name__)
        return

    if not technical_username:
        _report_attribution_unavailable("not_configured", service)
        return

    _resolve_and_bind(technical_username, service, derive_source=derive_source)


def declare_usage_unattributed() -> None:
    """Declare that this request deliberately has no accounting identity.

    Says: *no verified accounting identity exists here, and this is
    expected rather than a failure.* A runtime no-op by design - nothing is
    derived, nothing is bound, no request state is touched and, crucially,
    **no diagnostic is emitted**. That silence is what separates a
    deliberate absence from every runtime degradation path, so this
    operation does not pass through the shared failure tail.

    Its value is not runtime behaviour but that the absence becomes a fact
    in the code: a reader, a grep or a reviewer auditing which flows
    attribute can tell *deliberately unattributed* from *someone forgot*.

    Never raises, and is safe to call outside a request context.
    """
    return None
