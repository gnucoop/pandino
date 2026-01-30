from typing import Any
import json


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
    # (helps if the model returns extra wrapper text)
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

    - PandasAIEngine (legacy): may return arbitrary python objects -> keep as-is.
    - SmolagentsEngine (contract): may return dict with "kind" OR a string that contains
      a JSON contract payload (sometimes escaped). In that case, parse and return dict.
    """
    # Already contract dict
    if isinstance(raw_output, dict) and "kind" in raw_output:
        return raw_output

    # Contract JSON carried as string (common with LLM outputs)
    if isinstance(raw_output, str):
        parsed = _try_parse_contract_payload(raw_output)
        if parsed is not None:
            return parsed

    return raw_output