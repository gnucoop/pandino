
from typing import Any

import pandas as pd

from datachat.engine_interface import DataChatEngine
from datachat.pandasai_engine import PandasAIEngine
from datachat.smolagents_engine import SmolagentsEngine


def create_engine(
    *,
    engine_type: str,
    api_key: str,
    user_name: str,
    llm: Any,
    data: pd.DataFrame,
    open_charts: bool = False,
) -> DataChatEngine:
    """
    Create a DataChatEngine instance.

    AS-IS: only 'pandasai' is supported.
    Future: add 'smolagents' here without touching endpoints or stores.
    """
    normalized = (engine_type or "").lower().strip()
    if normalized in ("pandasai", ""):
        return PandasAIEngine(
            api_key=api_key,
            user_name=user_name,
            llm=llm,
            data=data,
            open_charts=open_charts,
        )

    if normalized == "smolagents":
        return SmolagentsEngine(
            api_key=api_key,
            user_name=user_name,
            llm=llm,
            data=data,
        )

    raise ValueError(f"Unsupported engine_type: {engine_type!r}")
