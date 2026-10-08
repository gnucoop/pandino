"""Request-local provider consumption performed inside DataChat tools.

A tool that calls a provider model itself - outside the CodeAgent run, whose
own token usage the route already records - reports what that call consumed
here. The route persists each contribution as an additional Usage row after
the engine run. Tools report consumption facts only: user, service, request
and row identity belong to the route and the Usage boundary.

State lives on ``flask.g``, so it ends with the request. Outside an application
context recording is a no-op and lookup returns an empty list.
"""

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "ProviderContribution",
    "get_provider_contributions",
    "record_provider_contribution",
]

_G_CONTRIBUTIONS_ATTR = "_maui_datachat_provider_contributions"


@dataclass(frozen=True)
class ProviderContribution:
    provider: str
    model: str
    token_input: int
    token_output: int


def _contributions(create: bool) -> Optional[list[ProviderContribution]]:
    from flask import g, has_app_context  # noqa: PLC0415

    if not has_app_context():
        return None
    contributions = getattr(g, _G_CONTRIBUTIONS_ATTR, None)
    if contributions is None and create:
        contributions = []
        setattr(g, _G_CONTRIBUTIONS_ATTR, contributions)
    return contributions


def record_provider_contribution(
    *, provider: str, model: str, token_input: int, token_output: int
) -> None:
    """Append one provider call's token consumption, in production order."""
    contributions = _contributions(create=True)
    if contributions is not None:
        contributions.append(
            ProviderContribution(
                provider=provider,
                model=model,
                token_input=token_input,
                token_output=token_output,
            )
        )


def get_provider_contributions() -> list[ProviderContribution]:
    """The contributions recorded during this request, oldest first."""
    return list(_contributions(create=False) or [])
