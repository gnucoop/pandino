import math
import os
from typing import Any

import pandas as pd

from file_manager import fileToBase64, isImageFilePath

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def replace_nan(data: Any) -> Any:
    """
    Recursively replace NaN with None inside nested dict/list structures.
    This is needed because JSON serialization doesn't handle NaN cleanly.
    """
    if isinstance(data, dict):
        return {k: replace_nan(v) for k, v in data.items()}
    if isinstance(data, list):
        return [replace_nan(item) for item in data]
    if isinstance(data, float) and math.isnan(data):
        return None
    return data


def _is_json_scalar(value: Any) -> bool:
    """Return True if the value is safely JSON-serializable as a scalar."""
    return value is None or isinstance(value, (str, int, float, bool))


def _sanitize_table_records(
    data: list[Any],
    *,
    max_rows: int = 50,
    max_columns: int = 10,
) -> list[dict[str, Any]]:
    """
    Sanitize a list of records (expected list[dict]) to make it safe and stable for Dino.

    Why this exists:
    - LLMs may output nested dict/list values (or strings containing unescaped JSON).
    - Dino expects list-of-dicts and Maui must be able to JSON-serialize them reliably.
    - We keep behavior general (no dataset-specific exclusions).

    Policy:
    - Accept only dict rows; skip non-dict rows.
    - Limit rows/columns (stability + payload size).
    - For each cell:
        - keep JSON scalars as-is
        - replace everything else with a compact string representation
          (this avoids nested JSON structures that can break downstream).
    """
    sanitized: list[dict[str, Any]] = []

    for row in data[:max_rows]:
        if not isinstance(row, dict):
            continue

        clean_row: dict[str, Any] = {}
        for k, v in row.items():
            if len(clean_row) >= max_columns:
                break

            # Normalize keys to string (defensive)
            key = str(k)

            if _is_json_scalar(v):
                clean_row[key] = v
            else:
                # Avoid nested dict/list/objects reaching Dino:
                # convert to string so JSON remains valid and predictable.
                clean_row[key] = str(v)

        sanitized.append(clean_row)

    return sanitized


# ---------------------------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------------------------


def normalize_datachat_response(response: Any) -> dict[str, Any]:
    """
    Normalize outputs into the *exact* "response_dict" structure expected by Dino (AS-IS).

    Supported legacy outputs (PandasAI):
      - list -> DataFrame -> dataframe records
      - DataFrame -> dataframe records
      - dict -> dict + {"type":"dict"}
      - other -> {"type": <pytype>, "value": <str>}

    Supported contract outputs (new engines):
      - {"kind":"text", ...}
      - {"kind":"error", ...}
      - {"kind":"image_path", ...}
      - {"kind":"table", "data": ...}

    Important: This function is intentionally defensive. It MUST:
      - never return malformed JSON structures
      - keep Dino compatibility
      - avoid dataset-specific assumptions
    """

    # -----------------------------------------------------------------------
    # Contract mode (new engines): dict with "kind"
    # -----------------------------------------------------------------------
    if isinstance(response, dict) and "kind" in response:
        kind = str(response.get("kind") or "").strip().lower()

        if kind == "text":
            text = response.get("text", "")
            return {"type": "str", "value": str(text)}

        if kind == "error":
            message = response.get("message", "")
            return {"type": "str", "value": str(message)}

        if kind == "image_path":
            path = response.get("path")
            if isinstance(path, str) and isImageFilePath(path):
                return {"type": "image", "value": fileToBase64(path)}
            return {"type": "str", "value": str(path)}

        if kind == "table":
            data = response.get("data")

            # Accept pandas DataFrame directly
            if isinstance(data, pd.DataFrame):
                records = data.to_dict(orient="records")
                return {"type": "dataframe", "value": replace_nan(records)}

            # Preferred: list-of-dicts (records)
            if isinstance(data, list):
                records = _sanitize_table_records(data)
                return {"type": "dataframe", "value": replace_nan(records)}

            # Rare: dict (already structured). Keep as dict to avoid guessing shape.
            if isinstance(data, dict):
                cleaned = replace_nan(data)
                if isinstance(cleaned, dict):
                    cleaned.update({"type": "dict"})
                    return cleaned
                return {"type": "dict", "value": cleaned}

            # Fallback: anything else as string
            return {"type": "str", "value": str(data)}

        # Unknown "kind": conservative fallback (dict payload)
        cleaned = replace_nan(response)
        if isinstance(cleaned, dict):
            cleaned.update({"type": "dict"})
            return cleaned
        return {"type": "dict", "value": cleaned}

    # -----------------------------------------------------------------------
    # Legacy mode (PandasAI / old behavior)
    # -----------------------------------------------------------------------

    # 1) list -> DataFrame (attempt)
    if isinstance(response, list):
        try:
            response = pd.DataFrame(response)
        except Exception as e:
            raise RuntimeError(f"Failed to convert list to DataFrame: {str(e)}") from e

    # 2) DataFrame -> dataframe records
    if isinstance(response, pd.DataFrame):
        return {"type": "dataframe", "value": replace_nan(response.to_dict(orient="records"))}

    # 3) dict -> dict + type
    if isinstance(response, dict):
        response_dict = replace_nan(response)
        if isinstance(response_dict, dict):
            response_dict.update({"type": "dict"})
            return response_dict
        return {"type": "dict", "value": response_dict}

    # 4) fallback -> string (AS-IS compatibility)
    response_dict: dict[str, Any] = {"type": type(response).__name__, "value": str(response)}

    # Keep the AS-IS "strange branches" without changing behavior now.
    if response_dict.get("value"):
        # NOTE: This condition is likely dead in current AS-IS, but kept for parity.
        if response_dict.get("type") == "string" and "plot" in response_dict:
            plot_path = response_dict.get("plot")
            if isinstance(plot_path, str) and os.path.exists(plot_path):
                response_dict["type"] = "text_and_image"
                response_dict["image"] = fileToBase64(plot_path)
                del response_dict["plot"]

        elif isinstance(response_dict.get("value"), str):
            v = response_dict["value"].strip()

            # rimuove eventuali quote esterne: "'...png'" o "\"...png\""
            if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                v = v[1:-1].strip()

            # 1) path così com’è
            if isImageFilePath(v):
                response_dict["type"] = "image"
                response_dict["value"] = fileToBase64(v)

            else:
                # 2) path assoluto (risolve problemi di cwd)
                abs_path = os.path.abspath(v)
                if isImageFilePath(abs_path):
                    response_dict["type"] = "image"
                    response_dict["value"] = fileToBase64(abs_path)
                else:
                    # niente conversione: resta stringa
                    response_dict["type"] = "str"
                    response_dict["value"] = v

    return response_dict