from typing import Any
import json
from contextvars import ContextVar


_adapter_fallback_used_ctx: ContextVar[bool] = ContextVar(
    "adapter_fallback_used",
    default=False,
)


def consume_adapter_fallback_used() -> bool:
    """
    Return and reset the per-request adapter fallback flag.

    This flag is set to True when adapt_engine_output had to coerce a non-contract
    output into the DataChat contract.
    """
    used = _adapter_fallback_used_ctx.get()
    _adapter_fallback_used_ctx.set(False)
    return bool(used)


def _try_parse_contract_payload(s: str) -> dict[str, Any] | None:
    """
    Try to parse a DataChat contract payload from a string.

    Handles:
    1) Proper JSON dict: {"kind":"text",...}
    2) Escaped JSON dict: {\"kind\":\"text\",...}   <-- common LLM mistake
    3) Double-encoded JSON string: "\"{...}\""
    """
    s = (s or "").strip()
    if not s:
        return None

    # (1) direct JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "kind" in obj:
            return obj
    except Exception:
        pass

    # (3) double-encoded JSON string -> decode once, then retry
    try:
        unwrapped = json.loads(s)
        if isinstance(unwrapped, str):
            return _try_parse_contract_payload(unwrapped)
    except Exception:
        pass

    # (2) escaped JSON object: {\"kind\":\"text\"...}
    # Heuristic: starts like an object and contains escaped quotes
    if s.startswith("{") and '\\"' in s:
        try:
            fixed = s.replace('\\"', '"')
            obj = json.loads(fixed)
            if isinstance(obj, dict) and "kind" in obj:
                return obj
        except Exception:
            pass

    # Conservative extraction: take substring between first { and last }
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1].strip()

        # try direct
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "kind" in obj:
                return obj
        except Exception:
            pass

        # try escaped
        if candidate.startswith("{") and '\\"' in candidate:
            try:
                fixed = candidate.replace('\\"', '"')
                obj = json.loads(fixed)
                if isinstance(obj, dict) and "kind" in obj:
                    return obj
            except Exception:
                pass

    return None


def adapt_engine_output(raw_output: Any) -> Any:
    """
    Adapter between engine-specific raw outputs and DataChat internal contract outputs.

    - If it is already a contract dict (has 'kind'), return it.
    - If it's a string containing a contract payload, parse it and return dict.
    - Otherwise: coerce common non-contract outputs into the contract:
        * list[dict] -> {"kind":"table","data":[...]}
        * str -> {"kind":"text","text":"...","format":"plain"}
        * list[str|int|float|bool|None] -> table with single column 'value'
    """
    _adapter_fallback_used_ctx.set(False)

    # 1) Already contract dict
    if isinstance(raw_output, dict) and "kind" in raw_output:
        return raw_output

    # 2) Contract JSON carried as string (common with LLM outputs)
    if isinstance(raw_output, str):
        s = raw_output.strip()

        parsed = _try_parse_contract_payload(s)
        if parsed is not None:
            return parsed

        # NEW: if it looks like an image filepath, return image_path contract
        s_low = s.lower()
        if s_low.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            _adapter_fallback_used_ctx.set(True)
            return {"kind": "image_path", "path": s}

        # Plain string -> contract text
        _adapter_fallback_used_ctx.set(True)
        return {"kind": "text", "text": s, "format": "plain"}

    # 3) Common legacy/tool shape: list of dicts -> contract table
    if isinstance(raw_output, list):
        if all(isinstance(item, dict) for item in raw_output):
            _adapter_fallback_used_ctx.set(True)
            return {"kind": "table", "data": raw_output}

        # list of scalars -> table with one column
        if all(
            (item is None) or isinstance(item, (str, int, float, bool))
            for item in raw_output
        ):
            _adapter_fallback_used_ctx.set(True)
            return {"kind": "table", "data": [{"value": v} for v in raw_output]}

        # unknown list content -> fallback text
        _adapter_fallback_used_ctx.set(True)
        return {"kind": "text", "text": str(raw_output), "format": "plain"}

    # 4) Anything else: keep as-is (or optionally coerce to text)
    _adapter_fallback_used_ctx.set(True)
    return raw_output
