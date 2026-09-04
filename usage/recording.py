# usage/recording.py
"""Public adoption boundary for recording explicit provider consumption.

This is the surface a Maui flow uses to say *what it consumed*. It is not a
generic utility: these are public operations of the Usage subsystem, and
everything they hide is deliberately not an adopter concern.

There are two consumption shapes, distinguished by who resolves the money::

    record_token_consumption(
        user_id=...,
        provider=...,
        model=...,
        service=...,
        token_input=...,
        token_output=...,
    )

    record_resolved_consumption(
        user_id=...,
        provider=...,
        model=...,
        service=...,
        cost=...,
    )

An adopter supplies consumption facts only, and receives a plain ``bool``
saying whether a Usage row was persisted.

What this boundary owns, so no adopter has to
----------------------------------------------
* **Writer selection.** The token shape is priced by Maui, so
  ``infrastructure.database_pg.log_token_usage`` performs the pricing
  lookup; the resolved-cost shape carries its own money and is written by
  ``log_usage_with_resolved_cost``, which performs no pricing lookup at
  all. Which writer serves which consumption shape is an internal
  decision; an adopter never chooses one. The token columns that writer
  fills for a non-token row are a storage convention and stay behind this
  boundary.
* **The Usage row id.** The id is received here, used here, and never
  returned. A caller cannot hold one, so it cannot forget to do anything
  with one.
* **Row-id registration, and therefore duration linkage.** The id is
  handed to ``usage.request_state.set_usage_log_id``, which both
  registers it for end-of-request duration finalization *and* keeps the
  latest-id compatibility slot current. Registration happens here, so
  ``logs.duration_ms`` does not depend on adopter bookkeeping.
* **Request correlation.** ``request_id`` comes from the ambient runtime
  logging context. There is deliberately no keyword for it, so an attempt
  to supply one fails as a ``TypeError`` at the call site.
* **Client source.** Derived (see :func:`_derive_source`), never passed.

Which operation to use follows from what the flow honestly holds. A flow
holding a provider-reported input/output token pair, for Maui to price,
records token consumption. A flow holding a monetary cost already resolved
- by the provider, or by the flow against a governed rate - records
resolved consumption, and supplies no quantity, because there is no
quantity it could state honestly for every provider it serves.

Failure model
-------------
Two kinds of failure, deliberately separated.

*Programmer-contract misuse* raises ``ValueError``: the public fields are
authored in a literal-ish form at the call site, so a wrong type or a
negative quantity is a defect to surface during development, not a runtime
condition to absorb. This mirrors the validation discipline of
``utils.operational_event``.

*Runtime accounting failure* never raises and never retries: the database
is unreachable, no pricing row exists, the source lookup fails. Accounting
is observation, so such a failure produces one safe diagnostic and
``False``, and the HTTP response it describes is unaffected. No retry,
because a failure between the INSERT and the commit acknowledgement cannot
be distinguished from a success, and retrying would risk double-counting
real money - the same reasoning recorded in
``usage.embedding_persistence``.

``token_input=0, token_output=0`` and ``cost=0`` are accepted and record a
row: a zero quantity is a real observation, not a missing one. Whether a
zero-consumption call *should* produce a row is a flow-level policy
question that current adopters answer inconsistently, so any such guard
stays at the call site rather than being decided here.

Request scope
-------------
Recording is request-scoped: registration and attribution both read
``flask.g``. Called outside a request context, this operation degrades
like any other runtime failure - one diagnostic, ``False`` - rather than
raising.
"""

import logging

from infrastructure.database_pg import (
    get_user_by_id,
    get_user_by_username,
    log_token_usage,
    log_usage_with_resolved_cost,
)
from utils.logging_config import get_request_id
from usage.attribution_state import get_usage_attribution
from usage.request_state import set_usage_log_id

logger = logging.getLogger(__name__)

__all__ = ["record_token_consumption", "record_resolved_consumption"]


def _require_int(name: str, value) -> int:
    """Require a plain int. ``bool`` is rejected: it is an ``int`` subclass,
    and ``True`` reaching a user id or a token count is a defect, not a 1."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _require_non_negative_int(name: str, value) -> int:
    """Require a non-negative int quantity. Zero is valid."""
    quantity = _require_int(name, value)
    if quantity < 0:
        raise ValueError(f"{name} must be non-negative, got {quantity}")
    return quantity


def _require_non_negative_number(name: str, value) -> float:
    """Require a non-negative real number. Zero is valid.

    ``bool`` is rejected for the same reason as in :func:`_require_int`. An
    ``int`` is accepted and widened: a whole-number amount of money is a
    real amount.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return float(value)


def _require_non_empty_str(name: str, value) -> str:
    """Require a str carrying something other than whitespace.

    Whitespace is rejected as well as emptiness: ``service`` is the key the
    admin Usage surfaces aggregate on, so a blank one silently fragments
    reporting rather than failing.
    """
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty str")
    return value


def _derive_source(user_id: int) -> "str | None":
    """Resolve the client source to persist for ``user_id``.

    Two ordered paths, both reading already-persisted identity rather than
    inferring one:

    1. **Bound Usage attribution**, when it exists *and* describes this same
       user. Attribution is the request's own answer to "who owns this
       consumption", so when it is present and matches, its ``source`` is
       used directly and no user lookup is needed.
    2. **The user's persisted row**, otherwise. ``get_user_by_id`` does not
       select the ``client`` column, so derivation resolves the username it
       does return and then reads the complete row through
       ``get_user_by_username``, which carries ``client``.

    ``None`` is a legitimate result, not a failure: a user may genuinely
    carry no client. A *failed* lookup is different and is left to the
    caller's guard, which treats it as a runtime accounting failure rather
    than fabricating an absence of client.
    """
    attribution = get_usage_attribution()
    if attribution is not None and attribution.user_id == user_id:
        return attribution.source

    user = get_user_by_id(user_id)
    if not user:
        raise LookupError(f"no user row for user_id={user_id}")

    username = user.get("username")
    if not username:
        raise LookupError(f"user row for user_id={user_id} carries no username")

    full_user = get_user_by_username(username)
    if not full_user:
        raise LookupError(f"user row for user_id={user_id} vanished during lookup")

    # ``get_user_by_username`` declares a loose ``str | int`` value type for
    # every column; ``client`` is a text column, so this narrows to what the
    # writer's ``source`` parameter accepts without altering a real value.
    client = full_user.get("client")
    return client if isinstance(client, str) else None


def record_token_consumption(
    *,
    user_id: int,
    provider: str,
    model: str,
    service: str,
    token_input: int,
    token_output: int,
) -> bool:
    """Record one token-based provider consumption as a Usage row.

    :param user_id: resolved Maui user id owning the consumption.
    :param provider: provider identity, as Maui's configuration spells it.
    :param model: model the consumption was billed against.
    :param service: HTTP endpoint that produced it, e.g. ``"/prompt.txt"``.
        Supplied explicitly because it is not derivable: one route may
        record more than one service literal.
    :param token_input: input tokens consumed. ``0`` is valid.
    :param token_output: output tokens produced. ``0`` is valid.
    :return: ``True`` when a Usage row was persisted, ``False`` when a
        runtime accounting failure prevented it.
    :raises ValueError: on programmer-contract misuse of any public field.

    There is no ``request_id`` and no ``source`` parameter, by design: both
    are derived internally, and their absence from the signature is what
    makes supplying them impossible rather than merely discouraged. No
    Usage row id is returned, ever.
    """
    # Validation runs before the fail-open guard, so misuse raises rather
    # than being absorbed as an accounting failure.
    user_id = _require_int("user_id", user_id)
    provider = _require_non_empty_str("provider", provider)
    model = _require_non_empty_str("model", model)
    service = _require_non_empty_str("service", service)
    token_input = _require_non_negative_int("token_input", token_input)
    token_output = _require_non_negative_int("token_output", token_output)

    try:
        request_id = get_request_id()
        source = _derive_source(user_id)

        log_id = log_token_usage(
            user_id=user_id,
            token_input=token_input,
            token_output=token_output,
            model=model,
            provider=provider,
            service=service,
            request_id=request_id,
            source=source,
        )

        # Internal bookkeeping, never the adopter's: registers the row for
        # end-of-request duration finalization and keeps the latest-id
        # compatibility slot current.
        set_usage_log_id(log_id)
        return True
    except Exception as exc:  # noqa: BLE001 - accounting must stay fail-open
        # One diagnostic, naming only configuration identities, the
        # endpoint, the correlation id and the failure type. Never the
        # prompt, the completion, the user identity or the exception text,
        # any of which can carry request content.
        logger.warning(
            "event=usage_token_recording_failed service=%s provider=%s "
            "model=%s error_type=%s",
            service,
            provider,
            model,
            type(exc).__name__,
        )
        return False


def record_resolved_consumption(
    *,
    user_id: int,
    provider: str,
    model: str,
    service: str,
    cost: float,
) -> bool:
    """Record one provider consumption whose monetary cost is already resolved.

    :param user_id: resolved Maui user id owning the consumption.
    :param provider: provider identity, as Maui's configuration spells it.
    :param model: model the consumption was billed against.
    :param service: HTTP endpoint that produced it, e.g. ``"/transcribe"``.
        Supplied explicitly because it is not derivable: one route may
        record more than one service literal.
    :param cost: the resolved monetary cost to persist. Non-negative;
        ``0`` is valid.
    :return: ``True`` when a Usage row was persisted, ``False`` when a
        runtime accounting failure prevented it.
    :raises ValueError: on programmer-contract misuse of any public field.

    No pricing lookup happens: the supplied cost is authoritative. As with
    the token shape, ``request_id`` and ``source`` are derived rather than
    accepted, and no Usage row id is returned, ever.
    """
    user_id = _require_int("user_id", user_id)
    provider = _require_non_empty_str("provider", provider)
    model = _require_non_empty_str("model", model)
    service = _require_non_empty_str("service", service)
    cost = _require_non_negative_number("cost", cost)

    try:
        request_id = get_request_id()
        source = _derive_source(user_id)

        log_id = log_usage_with_resolved_cost(
            user_id=user_id,
            cost=cost,
            model=model,
            provider=provider,
            service=service,
            request_id=request_id,
            source=source,
        )

        set_usage_log_id(log_id)
        return True
    except Exception as exc:  # noqa: BLE001 - accounting must stay fail-open
        # Names only configuration identities, the endpoint and the failure
        # type. Never the cost, the user identity or the exception text.
        logger.warning(
            "event=usage_resolved_cost_recording_failed service=%s provider=%s "
            "model=%s error_type=%s",
            service,
            provider,
            model,
            type(exc).__name__,
        )
        return False
