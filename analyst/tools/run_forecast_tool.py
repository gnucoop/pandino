from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from smolagents import Tool

from analyst.tools.dataset_common import DatasetToolError, get_dataset, numeric_columns, tool_error

_MIN_POINTS = 4
_MAX_PERIODS = 36
# Tried in order when the date column is text, e.g. report_mensile."mese" = '202604'.
_DATE_FORMATS = (None, "%Y%m", "%Y-%m")


def _as_dates(series: pd.Series) -> Optional[pd.Series]:
    """The column as datetimes, or None when it does not read as dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    for fmt in _DATE_FORMATS:
        try:
            parsed = pd.to_datetime(series.astype(str), format=fmt, errors="coerce")
        except (TypeError, ValueError):
            continue
        if parsed.notna().all():
            return parsed
    return None


def _label(period: Any) -> str:
    return period.date().isoformat() if isinstance(period, pd.Timestamp) else str(period)


def _round(value: float) -> float:
    return round(float(value), 4)


class RunForecastTool(Tool):
    """
    A linear-trend forecast of a value summed per period, on a loaded dataset.

    The value is summed per date, a linear trend is fitted (np.polyfit), and its error is measured as the
    MAPE of a fit on the first 80% of the points over the last 20%.
    """

    name = "run_forecast"
    description = (
        "Forecast value_column, summed per date_column, with a linear trend over a loaded "
        "dataset (at least 4 periods). Returns the trend slope, the next `periods` values "
        "(dated when the period can be inferred), the last 4-period moving average and the "
        "MAPE of the trend on the last 20% of the history. Exclude partial periods first."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "dataset": {"type": "string", "description": "Name of a dataset loaded with load_dataset."},
        "date_column": {
            "type": "string",
            "description": "The period column (dates, or text like '202604' / '2026-04').",
        },
        "value_column": {"type": "string", "description": "The numeric column to forecast."},
        "periods": {
            "type": "integer",
            "description": f"Periods to forecast (default 12, at most {_MAX_PERIODS}).",
            "nullable": True,
        },
    }

    def __init__(self, datasets: dict[str, pd.DataFrame]) -> None:
        super().__init__()
        self._datasets = datasets

    def forward(
        self, dataset: str, date_column: str, value_column: str, periods: Optional[int] = None
    ) -> dict[str, Any]:
        try:
            horizon = max(1, min(int(periods or 12), _MAX_PERIODS))
            frame = get_dataset(self._datasets, dataset)
            (value,) = numeric_columns(frame, [str(value_column or "").strip()])
            if date_column not in frame.columns:
                raise DatasetToolError(f"Unknown column {date_column!r}.", "INVALID_COLUMN")

            dates = _as_dates(frame[date_column])
            keys = dates if dates is not None else frame[date_column]
            series = frame[value].groupby(keys).sum().sort_index()
            if len(series) < _MIN_POINTS:
                raise DatasetToolError(
                    f"Too few periods ({len(series)}): at least {_MIN_POINTS} are needed."
                )

            y = series.to_numpy(dtype=float)
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            future = slope * np.arange(len(y), len(y) + horizon) + intercept

            labels: list[str] = [f"t+{i + 1}" for i in range(horizon)]
            freq = pd.infer_freq(series.index) if dates is not None and len(series) >= 3 else None
            if freq:
                labels = [
                    _label(d)
                    for d in pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]
                ]

            # Fit on the first 80%, measure on the rest: two points to fit, one to measure.
            split = int(len(y) * 0.8)
            mape = None
            if split >= 2 and len(y) - split >= 1:
                s, i = np.polyfit(x[:split], y[:split], 1)
                actual = y[split:]
                fitted = s * x[split:] + i
                nonzero = actual != 0
                if nonzero.any():
                    mape = _round(np.mean(np.abs((actual[nonzero] - fitted[nonzero]) / actual[nonzero])) * 100)

            return {
                "kind": "forecast",
                "periods_used": len(y),
                "first_period": _label(series.index[0]),
                "last_period": _label(series.index[-1]),
                "last_value": _round(y[-1]),
                "slope_per_period": _round(slope),
                "trend": "up" if slope > 0 else "down" if slope < 0 else "flat",
                "moving_average_4": _round(y[-4:].mean()),
                "forecast": [{"period": p, "value": _round(v)} for p, v in zip(labels, future)],
                "mape_percent": mape,
            }
        except Exception as e:
            return tool_error(e, self.name)
