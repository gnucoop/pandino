from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from smolagents import Tool

from analyst.tools.dataset_common import DatasetToolError, get_dataset, numeric_columns, tool_error

_MODELS = {
    "linear": lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=1.0),
    "lasso": lambda: Lasso(alpha=0.1),
    "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
}
_MIN_ROWS = 10


def _round(value: float) -> float:
    return round(float(value), 4)


class RunRegressionTool(Tool):
    """
    A regression of one numeric column on others, on a loaded dataset.

    An 80/20 split with a fixed seed, test-set metrics. 
    Only the metrics and the coefficients (or feature importances) come back.
    """

    name = "run_regression"
    description = (
        "Fit a regression of target on features over a loaded dataset (80/20 train/test "
        "split). method: 'linear' (default), 'ridge', 'lasso' or 'random_forest'. Returns "
        "test-set r2, rmse and mae, plus coefficients and intercept (linear models) or "
        "feature importances (random_forest)."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "dataset": {"type": "string", "description": "Name of a dataset loaded with load_dataset."},
        "target": {"type": "string", "description": "The numeric column to predict."},
        "features": {"type": "string", "description": "Comma-separated numeric predictor columns."},
        "method": {
            "type": "string",
            "description": "'linear' (default), 'ridge', 'lasso' or 'random_forest'.",
            "nullable": True,
        },
    }

    def __init__(self, datasets: dict[str, pd.DataFrame]) -> None:
        super().__init__()
        self._datasets = datasets

    def forward(
        self, dataset: str, target: str, features: str, method: Optional[str] = None
    ) -> dict[str, Any]:
        try:
            method = str(method or "linear").strip().lower()
            if method not in _MODELS:
                raise DatasetToolError(f"method must be one of: {', '.join(_MODELS)}.")

            frame = get_dataset(self._datasets, dataset)
            (target_column,) = numeric_columns(frame, [str(target or "").strip()])
            feature_columns = [c for c in numeric_columns(frame, features) if c != target_column]
            if not feature_columns:
                raise DatasetToolError("features must name at least one column other than target.")

            data = frame[[*feature_columns, target_column]].dropna()
            if len(data) < _MIN_ROWS:
                raise DatasetToolError(
                    f"Too few complete rows ({len(data)}): at least {_MIN_ROWS} are needed."
                )

            x_train, x_test, y_train, y_test = train_test_split(
                data[feature_columns], data[target_column], test_size=0.2, random_state=42
            )
            model = _MODELS[method]()
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)

            result: dict[str, Any] = {
                "kind": "regression",
                "method": method,
                "target": target_column,
                "rows": len(data),
                "r2": _round(r2_score(y_test, predicted)),
                "rmse": _round(np.sqrt(mean_squared_error(y_test, predicted))),
                "mae": _round(mean_absolute_error(y_test, predicted)),
            }
            if method == "random_forest":
                result["feature_importance"] = {
                    c: _round(v) for c, v in zip(feature_columns, model.feature_importances_)
                }
            else:
                result["coefficients"] = {c: _round(v) for c, v in zip(feature_columns, model.coef_)}
                result["intercept"] = _round(model.intercept_)
            return result
        except Exception as e:
            return tool_error(e, self.name)
