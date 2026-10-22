"""
model_options.py
----------------

How the analyst's agents call their model.

A provider call can hang: LiteLLM's default timeout is 600 s, and a timed out
call fails the agent run, so a whole task attempt (and its steps) was lost to
one stuck request. The analyst's calls are cut at MODEL_TIMEOUT_S and re-sent by
LiteLLM itself up to MODEL_RETRIES times (timeouts, connection and API errors),
so the agent run goes on. The interviewer and DataChat keep LiteLLM's defaults.
"""

from __future__ import annotations

from typing import Any

MODEL_TIMEOUT_S = 180
MODEL_RETRIES = 2


class AnalystModelOptions:
    """Mixin for the analyst engines: the options their model is built with."""

    def _model_options(self) -> dict[str, Any]:
        return {"timeout": MODEL_TIMEOUT_S, "num_retries": MODEL_RETRIES}
