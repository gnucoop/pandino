# utils/agent_serialization.py
from __future__ import annotations
from typing import Any, Dict, List
import ast
import json

def _try_parse_vectors_from_string(s: str) -> list[dict]:
    """
    Handles cases where observations are a string:
    'Execution logs:\n{\'vectors\': [...], \'used\': {...}}\nLast output...'
    Attempts to safely extract the dict using ast.literal_eval.
    """
    if not isinstance(s, str) or "{'vectors':" not in s:
        return []
    try:
        # isolate the block that starts with {'vectors':
        start = s.find("{'vectors':")
        # end before "\nLast output" or at the end of the string
        end = s.find("\nLast output", start)
        fragment = s[start:] if end == -1 else s[start:end]
        obs_dict = ast.literal_eval(fragment)
        return obs_dict.get("vectors", []) if isinstance(obs_dict, dict) else []
    except Exception:
        return []

def _extract_vectors_from_steps(steps):
    """
    Extracts all vectors present in the RunResult steps.
    Deduplicates results based on the text found in metadata['text'].
    """
    vectors = []
    seen = set()  # keeps track of already added texts to avoid duplicates

    for step in steps or []:
        obs = step.get("observations")

        # Case 1: JSON dict with 'vectors' key
        if isinstance(obs, dict) and "vectors" in obs:
            new_vectors = obs.get("vectors", [])

        # Case 2: string (Execution logs as text)
        elif isinstance(obs, str):
            new_vectors = _try_parse_vectors_from_string(obs)

        # Case 3: no vector available
        else:
            new_vectors = []

        # Iterate over the newly found vectors
        for v in new_vectors:
            metadata = v.get("metadata", {}) or {}
            text = metadata.get("text", "").strip()

            # Avoid duplicates based on textual content
            if text and text not in seen:
                seen.add(text)
                vectors.append({
                    "similarity": float(v.get("similarity", 0.0)),
                    "metadata": metadata
                })

    return vectors

def _extract_simple_tool_calls(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for step in steps or []:
        for tc in step.get("tool_calls", []) or []:
            fn = (tc.get("function") or {})
            calls.append({
                "tool_name": fn.get("name"),
                "arguments": fn.get("arguments"),
                "duration_ms": round(float(step.get("timing", {}).get("duration", 0.0)) * 1000, 2) if step.get("timing") else None,
            })
    return calls

def serialize_runresult(result: Any) -> Dict[str, Any]:
    """
    Serializes a RunResult (smolagents) into a compact, stable JSON for /agentchat.
    It has no Flask dependencies, so it remains easily testable.
    """
    steps: List[Dict[str, Any]] = getattr(result, "steps", []) or []
    timing = getattr(result, "timing", None)
    token_usage = getattr(result, "token_usage", None)

    # answer + follow_ups
    answer, follow_ups = "", []
    out = getattr(result, "output", None)
    if isinstance(out, dict):
        answer = str(out.get("answer", "")).strip()
        fu = out.get("follow_ups", []) or []
        if isinstance(fu, list):
            follow_ups = [str(x).strip() for x in fu if str(x).strip()]
    elif isinstance(out, str):
        try:
            parsed = json.loads(out)
            answer = str(parsed.get("answer", "")).strip()
            fu = parsed.get("follow_ups", []) or []
            if isinstance(fu, list):
                follow_ups = [str(x).strip() for x in fu if str(x).strip()]
        except Exception:
            pass


    # vectors + tool_calls
    vectors = _extract_vectors_from_steps(steps)
    tool_calls = _extract_simple_tool_calls(steps)

    metrics = {
        "duration_ms": round(float(getattr(timing, "duration", 0.0)) * 1000, 2) if timing else None,
        "token_usage": {
            "input": getattr(token_usage, "input_tokens", None),
            "output": getattr(token_usage, "output_tokens", None),
            "total": getattr(token_usage, "total_tokens", None),
        },
    }

    debug = {
        "state": getattr(result, "state", None),
        "steps_count": len(steps),
    }

    return {
        "answer": answer,
        "follow_ups": follow_ups,
        "vectors": vectors,
        "tool_calls": tool_calls,
        "metrics": metrics,
        "debug": debug,
    }
