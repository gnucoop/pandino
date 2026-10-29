from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from smolagents import Tool

from analyst.tools.dataset_common import DatasetToolError, get_dataset, numeric_columns, tool_error
from analyst.values import to_native

_METHODS = ("isolation_forest", "zscore", "iqr")
_MAX_INDICES = 50
_MAX_SAMPLE_ROWS = 5
_MAX_SAMPLE_COLUMNS = 25


class DetectAnomaliesTool(Tool):
    """
    Anomalous rows of a loaded dataset, on one or more numeric columns.

    (Isolation forest with 10% contamination and a fixed seed, |z| > 3, 1.5 x
    IQR). Only counts, a few indices and a small sample come back.
    """

    name = "detect_anomalies"
    description = (
        "Find anomalous rows of a loaded dataset. method: 'isolation_forest' (default; "
        "multivariate, flags ~10% of rows), 'zscore' (|z| > 3 on any column) or 'iqr' "
        "(outside 1.5 x IQR on any column). Returns the count, the rate, up to 50 row "
        "indices, a 5-row sample and the column means of anomalous vs normal rows."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "dataset": {"type": "string", "description": "Name of a dataset loaded with load_dataset."},
        "columns": {"type": "string", "description": "Comma-separated numeric columns."},
        "method": {
            "type": "string",
            "description": "'isolation_forest' (default), 'zscore' or 'iqr'.",
            "nullable": True,
        },
    }

    def __init__(self, datasets: dict[str, pd.DataFrame]) -> None:
        super().__init__()
        self._datasets = datasets

    def forward(self, dataset: str, columns: str, method: Optional[str] = None) -> dict[str, Any]:
        try:
            method = str(method or "isolation_forest").strip().lower()
            if method not in _METHODS:
                raise DatasetToolError(f"method must be one of: {', '.join(_METHODS)}.")

            frame = get_dataset(self._datasets, dataset)
            cols = numeric_columns(frame, columns)
            data = frame[cols].dropna()
            if len(data) < 10:
                raise DatasetToolError(f"Too few complete rows ({len(data)}): at least 10 are needed.")

            values = data.to_numpy(dtype=float)
            if method == "isolation_forest":
                scaled = StandardScaler().fit_transform(values)
                model = IsolationForest(contamination=0.1, random_state=42, n_jobs=1)
                flags = model.fit_predict(scaled) == -1
            elif method == "zscore":
                std = values.std(axis=0)
                std[std == 0] = 1.0
                flags = (np.abs((values - values.mean(axis=0)) / std) > 3).any(axis=1)
            else:
                q1, q3 = np.percentile(values, [25, 75], axis=0)
                spread = 1.5 * (q3 - q1)
                flags = ((values < q1 - spread) | (values > q3 + spread)).any(axis=1)

            anomalies = data[flags]
            return {
                "kind": "anomalies",
                "method": method,
                "anomaly_count": int(flags.sum()),
                "total_records": len(data),
                "anomaly_rate": round(float(flags.mean()), 4),
                "anomaly_indices": to_native(anomalies.index[:_MAX_INDICES]),
                "sample": to_native(
                    frame.loc[anomalies.index[:_MAX_SAMPLE_ROWS]].iloc[:, :_MAX_SAMPLE_COLUMNS]
                ),
                "anomaly_means": to_native(anomalies.mean().round(4)),
                "normal_means": to_native(data[~flags].mean().round(4)),
            }
        except Exception as e:
            return tool_error(e, self.name)
