# utils/agent_serialization.py
from __future__ import annotations
from typing import Any, Dict, List
import ast

def _try_parse_vectors_from_string(s: str) -> list[dict]:
    """
    Gestisce casi di observations come stringa:
    'Execution logs:\n{\'vectors\': [...], \'used\': {...}}\nLast output...'
    Prova a estrarre il dict con ast.literal_eval in modo sicuro.
    """
    if not isinstance(s, str) or "{'vectors':" not in s:
        return []
    try:
        # isola il blocco che inizia con {'vectors':
        start = s.find("{'vectors':")
        # termina prima di "\nLast output" o alla fine stringa
        end = s.find("\nLast output", start)
        fragment = s[start:] if end == -1 else s[start:end]
        obs_dict = ast.literal_eval(fragment)
        return obs_dict.get("vectors", []) if isinstance(obs_dict, dict) else []
    except Exception:
        return []

def _extract_vectors_from_steps(steps):
    """
    Estrae tutti i vettori presenti negli step del RunResult.
    Deduplica i risultati in base al testo contenuto in metadata['text'].
    """
    vectors = []
    seen = set()  # tiene traccia dei testi già aggiunti per evitare duplicati

    for step in steps or []:
        obs = step.get("observations")

        # Caso 1: dict JSON con chiave 'vectors'
        if isinstance(obs, dict) and "vectors" in obs:
            new_vectors = obs.get("vectors", [])

        # Caso 2: stringa (Execution logs come testo)
        elif isinstance(obs, str):
            new_vectors = _try_parse_vectors_from_string(obs)

        # Caso 3: nessun vettore disponibile
        else:
            new_vectors = []

        # Itera sui nuovi vettori trovati
        for v in new_vectors:
            metadata = v.get("metadata", {}) or {}
            text = metadata.get("text", "").strip()

            # Evita duplicati basandosi sul contenuto testuale
            if text and text not in seen:
                seen.add(text)
                vectors.append({
                    "similarity": float(v.get("similarity", 0.0)),
                    "metadata": metadata
                })

    return vectors

# def _extract_vectors_from_steps(steps):
#     vectors = []
#     seen = set()  # per deduplicare sul contenuto testuale
    
#     for step in steps or []:
#         obs = step.get("observations")
#         if isinstance(obs, dict) and "vectors" in obs:
#             for v in obs.get("vectors", []):
#                 vectors.append({"similarity": float(v.get("similarity", 0.0)),
#                                 "metadata": v.get("metadata", {}) or {}})
#         elif isinstance(obs, str):
#             for v in _try_parse_vectors_from_string(obs):
#                 vectors.append({"similarity": float(v.get("similarity", 0.0)),
#                                 "metadata": v.get("metadata", {}) or {}})
#     return vectors

def _extract_simple_tool_calls(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for step in steps or []:
        for tc in step.get("tool_calls", []) or []:
            fn = (tc.get("function") or {})
            calls.append({
                "tool_name": fn.get("name"),
                "arguments": fn.get("arguments"),
                # opzionale: durata step come proxy della tool call
                "duration_ms": round(float(step.get("timing", {}).get("duration", 0.0)) * 1000, 2) if step.get("timing") else None,
            })
    return calls

def serialize_runresult(result: Any) -> Dict[str, Any]:
    """
    Serializza un RunResult (smolagents) in un JSON compatto e stabile per /compass/agentchat.
    Non ha dipendenze da Flask, così resta facilmente testabile.
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
