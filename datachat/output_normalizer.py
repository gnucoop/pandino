import logging
import math
import os
from typing import Any, Optional

import pandas as pd

from infrastructure.file_manager import fileToBase64, isImageFilePath

# Table responses are sent to the client as a *preview*: the first _PREVIEW_ROWS rows
# and _PREVIEW_COLUMNS columns, to keep the JSON payload small. Whenever a result
# exceeds either limit the full version is written to CSV and offered as a download,
# and the response says so explicitly (see _build_table_response).
_PREVIEW_ROWS = 20
_PREVIEW_COLUMNS = 10

_EXPORT_URL_PREFIX = "/datachat/export"

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
    max_rows: int = _PREVIEW_ROWS,
    max_columns: int = _PREVIEW_COLUMNS,
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


def _count_columns(records: list[Any]) -> int:
    """Number of distinct column names across all records."""
    keys: set[str] = set()
    for row in records:
        if isinstance(row, dict):
            keys.update(str(k) for k in row.keys())
    return len(keys)


def _build_table_response(
    records: list[Any],
    exporter: Any = None,
    hint: str = "export",
    note: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build the client response for a table result.

    `records` must be the *complete* result. The client only receives a preview
    (first _PREVIEW_ROWS rows / _PREVIEW_COLUMNS columns), so this is the one place
    that knows the real size — and therefore the only place that can both report it
    and write the full CSV before the rest is dropped.

    The "type"/"value" pair is unchanged for backwards compatibility; the counters and
    download fields are additive, so a client that ignores them renders exactly as before.
    """
    total_rows = len(records)
    total_columns = _count_columns(records)

    preview = replace_nan(_sanitize_table_records(records))
    truncated = total_rows > _PREVIEW_ROWS or total_columns > _PREVIEW_COLUMNS

    payload: dict[str, Any] = {
        "type": "dataframe",
        "value": preview,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "preview_rows": len(preview) if isinstance(preview, list) else 0,
        "truncated": truncated,
        "download_url": None,
        "download_filename": None,
    }

    # A tool may flag a caveat about its own result (e.g. rows it could not analyze).
    if note:
        payload["note"] = str(note)

    if not truncated or exporter is None or not hasattr(exporter, "register_export"):
        return payload

    # Export the full result. A failure here must never cost the user their answer,
    # so fall back to a plain (labelled) preview.
    try:
        token, download_filename = exporter.register_export(
            [row for row in records if isinstance(row, dict)],
            hint=hint,
        )
        payload["download_url"] = f"{_EXPORT_URL_PREFIX}/{token}"
        payload["download_filename"] = download_filename
    except Exception as e:
        logging.warning("[datachat][output_normalizer] full-result export failed: %s", e)

    return payload


# ---------------------------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------------------------


def normalize_datachat_response(response: Any, exporter: Any = None) -> dict[str, Any]:
    """
    Normalize outputs into the *exact* "response_dict" structure expected by Dino (AS-IS).

    `exporter` is optional and, when supplied, must expose
    `register_export(records, hint) -> (token, download_filename)` (the active engine).
    It is used to write the full CSV for table results that exceed the preview limits.
    Omitting it keeps the previous behaviour minus the download fields.

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
            # Be tolerant with common LLM variants: content/message -> text
            text = response.get("text")
            if text is None:
                text = response.get("content")
            if text is None:
                text = response.get("message")
            if text is None:
                text = ""

            payload: dict[str, Any] = {"type": "str", "value": str(text)}

            # A tool may attach a download (e.g. export_csv) to a plain text answer.
            download_url = response.get("download_url")
            if isinstance(download_url, str) and download_url:
                payload["download_url"] = download_url
                payload["download_filename"] = response.get("download_filename")
            return payload

        if kind == "error":
            message = response.get("message")
            if message is None:
                message = response.get("text")
            if message is None:
                message = response.get("content")
            if message is None:
                message = ""
            return {"type": "str", "value": str(message)}

        if kind == "image_path":
            path = response.get("path")
            if isinstance(path, str) and isImageFilePath(path):
                return {"type": "image", "value": fileToBase64(path)}
            return {"type": "str", "value": str(path)}

        if kind == "table":
            data = response.get("data")
            # Optional label a tool can set to name the downloaded file.
            hint = str(response.get("export_name") or "export")
            # Optional caveat a tool can attach to its own result.
            note = response.get("note")

            # Accept pandas DataFrame directly
            if isinstance(data, pd.DataFrame):
                return _build_table_response(
                    data.to_dict(orient="records"), exporter=exporter, hint=hint, note=note
                )

            # Preferred: list-of-dicts (records)
            if isinstance(data, list):
                return _build_table_response(
                    data, exporter=exporter, hint=hint, note=note
                )

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
    # Routed through the same preview/export path as the contract branch: otherwise a
    # legacy DataFrame would escape the row/column caps entirely.
    if isinstance(response, pd.DataFrame):
        return _build_table_response(
            response.to_dict(orient="records"), exporter=exporter
        )

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