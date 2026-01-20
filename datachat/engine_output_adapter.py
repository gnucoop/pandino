
from typing import Any


def adapt_engine_output(raw_output: Any) -> Any:
    """
    Adapter between engine-specific raw outputs and DataChat internal contract outputs.

    Today (AS-IS): PandasAIEngine returns arbitrary Python objects (legacy mode),
    so this function is identity.

    Future: SmolagentsEngine will return tool results / messages that we will convert
    into dict-based InternalOutput with a 'kind' key (contract mode).
    """
    return raw_output
