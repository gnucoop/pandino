"""
final_answer_contract.py
------------------------

The final-answer contract of a smolagents engine (SmolagentsEngineBase).

An engine run must end with exactly one JSON object carrying a "kind" field.
What each kind must contain is engine-specific — the interviewer answers with
questions and brief drafts — so the kinds are not fixed here: each engine
passes its own map of kind -> validator.
What is shared is everything around that map: pulling the object out of
whatever the model produced, normalizing the kind, and reporting a reason the
model can act on when the answer is turned down.

A validator receives the parsed payload and returns None when it is valid, or
an UPPER_SNAKE reason code when it is not.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Optional, Tuple

KindValidator = Callable[[dict[str, Any]], Optional[str]]
PayloadRepair = Callable[[dict[str, Any]], dict[str, Any]]


def extract_json_object(s: str) -> Optional[dict[str, Any]]:
    """
    Best-effort extraction of a JSON object from a string.
    - strict json.loads
    - if that fails, try substring between first '{' and last '}'.
    """
    s = (s or "").strip()
    if not s:
        return None

    # A) strict JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        # Sometimes LLM returns a JSON string containing JSON.
        if isinstance(obj, str):
            obj2 = json.loads(obj)
            if isinstance(obj2, dict):
                return obj2
    except Exception:
        pass

    # B) conservative extraction
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1].strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def normalize_kind(payload: Mapping[str, Any]) -> str:
    """The payload's kind, stripped and lowercased; "" when absent."""
    return str(payload.get("kind") or "").strip().lower()


# ----------------------------
# Validators shared across engines
# ----------------------------

def validate_text_or_message(payload: dict[str, Any]) -> Optional[str]:
    """kind="text" / kind="error": a non-blank "text" or "message"."""
    text_val = payload.get("text")
    msg_val = payload.get("message")
    if text_val is None and msg_val is None:
        return "MISSING_TEXT_OR_MESSAGE"
    if text_val is not None and not str(text_val).strip() and msg_val is None:
        return "EMPTY_TEXT"
    if msg_val is not None and not str(msg_val).strip() and text_val is None:
        return "EMPTY_MESSAGE"
    return None


def validate_table(payload: dict[str, Any]) -> Optional[str]:
    """kind="table": a "data" field (an empty list is valid here)."""
    if "data" not in payload:
        return "MISSING_DATA"
    return None


def unwrap_nested_table(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Fix common LLM mistake:
      {"kind":"table","data":{"kind":"table","data":[...]}}
    -> {"kind":"table","data":[...]}
    """
    try:
        if normalize_kind(payload) != "table":
            return payload

        data = payload.get("data")
        if isinstance(data, list):
            return payload

        if isinstance(data, dict):
            nested_data = data.get("data")
            if isinstance(nested_data, list):
                payload["data"] = nested_data
        return payload
    except Exception:
        return payload


def is_empty_table(payload: Optional[dict[str, Any]]) -> bool:
    """Whether a validated payload is a table that carries no rows."""
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    return isinstance(data, list) and not data


# ----------------------------
# Contract evaluation
# ----------------------------

def validate_contract_payload(
    payload: dict[str, Any],
    validators: Mapping[str, KindValidator],
) -> Tuple[bool, Optional[str], str]:
    """
    Returns:
      (passed, final_kind, reason)
    """
    kind = normalize_kind(payload)

    validator = validators.get(kind)
    if validator is None:
        return False, (kind or None), "INVALID_KIND"

    reason = validator(payload)
    if reason is not None:
        return False, kind, reason
    return True, kind, "OK"


def coerce_final_payload(
    output: Any,
    validators: Mapping[str, KindValidator],
    *,
    repair: Optional[PayloadRepair] = None,
) -> Tuple[Optional[dict[str, Any]], bool, Optional[str], str]:
    """
    Parse + repair + validate.

    :param output: whatever the agent handed to final_answer (dict or string).
    :param validators: the engine's kind -> validator map.
    :param repair: optional fix-up applied before validation (e.g. unwrapping
        a doubly nested table).
    :return: (payload_or_none, passed, final_kind, reason)
    """
    if isinstance(output, dict):
        candidate = output
    elif isinstance(output, str):
        candidate = extract_json_object(output)
        if candidate is None:
            return None, False, None, "NON_JSON_OR_NO_OBJECT"
    else:
        return None, False, None, "NON_JSON_OR_UNSUPPORTED_TYPE"

    if "kind" not in candidate:
        return None, False, None, "NO_KIND"

    if repair is not None:
        candidate = repair(candidate)
    passed, final_kind, reason = validate_contract_payload(candidate, validators)
    return candidate, passed, final_kind, reason
