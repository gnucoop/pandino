"""
metrics.py
----------

The tracked metrics of a run, and their comparison with the previous run of
the same brief.

The plan's metrics_config names output keys (validate_plan() guarantees
they exist): `metrics` are numbers shown as KPI cards, `categories` are lists
whose items, and item count, are compared between runs. Their values are read
from the task outputs here, not relayed by a model, and every completed run leaves them in its
run_metadata.json, which is what the next run of the brief compares against.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from analyst.values import is_dataset_reference, to_native

logger = logging.getLogger(__name__)

METADATA_FILE = "run_metadata.json"


def _find(outputs: Mapping[str, Any], key: str) -> Any:
    """The value under `key`, at the top level or one level down (digest's _deep_find)."""
    if key in outputs:
        return outputs[key]
    for value in outputs.values():
        if isinstance(value, Mapping) and key in value:
            return value[key]
    return None


def _table_rows(value: Any, datasets: Mapping[str, Any]) -> Any:
    """
    A dataset reference (values.dataset_reference) resolved to the rows of its
    table: the whole dataset while the run holds it, else the first rows saved
    with the reference. Anything else is returned as it is.
    """
    if not is_dataset_reference(value):
        return value
    frame = datasets.get(value["dataset"])
    return to_native(frame) if frame is not None else list(value["head"] or [])


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    # A metric saved as {"value": 123, ...} is common enough to accept.
    if isinstance(value, Mapping) and isinstance(value.get("value"), (int, float)):
        return value["value"]
    return None


def fill_metrics(
    metrics_config: Optional[Mapping[str, Any]],
    outputs: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
    datasets: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """
    {"metrics": {key: number}, "categories": {key: [items]}, "missing": [keys]}.

    :param overrides: values the report agent passed explicitly; they win.
    :param datasets: the run's datasets, to count a category saved as a dataset
        reference by its table's rows rather than by the reference's keys.
    """
    config = metrics_config or {}
    overrides = overrides or {}
    result: dict[str, Any] = {"metrics": {}, "categories": {}, "missing": []}

    for key in config.get("metrics") or {}:
        value = _number(overrides.get(key, _find(outputs, key)))
        if value is None:
            result["missing"].append(key)
        else:
            result["metrics"][key] = value

    for key in config.get("categories") or {}:
        value = _table_rows(overrides.get(key, _find(outputs, key)), datasets or {})
        if isinstance(value, Mapping):
            value = [{"key": k, "value": v} for k, v in value.items()]
        if isinstance(value, list):
            result["categories"][key] = value
        else:
            result["missing"].append(key)
    return result


def find_previous_run(reports_dir: Path, brief_id: int, current_run_id: str) -> Optional[dict[str, Any]]:
    """The metadata of the latest completed run of the brief, other than the current one."""
    latest = None
    for path in reports_dir.glob(f"*/{METADATA_FILE}"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("event=analyst_metadata_unreadable path=%s", path)
            continue
        if (
            meta.get("brief_id") == brief_id
            and meta.get("status") == "completed"
            and meta.get("run_id") != current_run_id
            and (latest is None or str(meta.get("finished_at")) > str(latest.get("finished_at")))
        ):
            latest = meta
    return latest


def _hashable(item: Any) -> str:
    return json.dumps(item, sort_keys=True, default=str)


def compute_delta(
    metrics_config: Mapping[str, Any],
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Metric changes and category additions/removals between two runs.

    `current` and `previous` are fill_metrics() results ({"metrics", "categories"}).
    """
    metrics: dict[str, Any] = {}
    for key, spec in (metrics_config.get("metrics") or {}).items():
        now = current.get("metrics", {}).get(key)
        before = previous.get("metrics", {}).get(key)
        if now is None or before is None:
            continue
        change = now - before
        metrics[key] = {
            "label": spec.get("label", key),
            "current": now,
            "previous": before,
            "change": round(change, 4),
            "percent": round(change / before * 100, 1) if before else None,
            # None when unchanged: neither better nor worse.
            "improved": None if change == 0 else (change > 0) == (spec.get("better", "higher") == "higher"),
        }

    categories: dict[str, Any] = {}
    for key, spec in (metrics_config.get("categories") or {}).items():
        now = {_hashable(i): i for i in current.get("categories", {}).get(key, [])}
        before = {_hashable(i): i for i in previous.get("categories", {}).get(key, [])}
        added = [now[h] for h in now if h not in before]
        removed = [before[h] for h in before if h not in now]
        if added or removed:
            categories[key] = {
                "label": spec.get("label", key),
                "count": len(now),
                "previous_count": len(before),
                "added": added[:5],
                "removed": removed[:5],
            }
    return {"metrics": metrics, "categories": categories}
