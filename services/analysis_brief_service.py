"""
analysis_brief_service.py
-------------------------

Persistence of the analysis briefs produced by the interviewer.

A brief reaches this service only after the user has approved it: nothing the
interviewer merely proposes is ever stored. This is also where the data
analyst will read briefs from. Flask-free, so both the HTTP layer and a
background agent can use it.
"""

from typing import Any, Optional, TypedDict

from infrastructure.database_pg import (
    get_analysis_brief,
    list_analysis_briefs,
    save_analysis_brief,
)


class SavedBrief(TypedDict):
    brief_id: int
    name: str


def save_approved_brief(
    *,
    user_email: str,
    brief: dict[str, Any],
    prompt_text: str,
    transcript: Optional[list[dict[str, Any]]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> SavedBrief:
    """
    Store a brief the user has approved.

    :param user_email: the user who approved it.
    :param brief: the validated structured brief.
    :param prompt_text: the prompt rendered from it for the data analyst.
    :param transcript: the interview that produced it, for audit.
    :param model: model that conducted the interview.
    :param provider: provider of that model.
    :return: the new brief id and its name.
    :raises RuntimeError: if the brief could not be stored.
    """
    name = str(brief.get("name") or "Untitled analysis").strip()
    try:
        brief_id = save_analysis_brief(
            user_email=user_email,
            name=name,
            brief=brief,
            prompt_text=prompt_text,
            transcript=transcript,
            model=model,
            provider=provider,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save analysis brief: {e}") from e
    return SavedBrief(brief_id=brief_id, name=name)


def get_brief(brief_id: int, user_email: Optional[str] = None) -> Optional[dict[str, Any]]:
    """
    Load one stored brief.

    :param brief_id: the brief id.
    :param user_email: when given, the brief must belong to this user.
    :return: the stored row (brief, prompt_text, transcript, status, ...), or None.
    :raises RuntimeError: if the database could not be read.
    """
    try:
        return get_analysis_brief(brief_id, user_email)
    except Exception as e:
        raise RuntimeError(f"Failed to load analysis brief {brief_id}: {e}") from e


def list_briefs(
    user_email: Optional[str] = None, status: Optional[str] = None, limit: int = 50
) -> list[dict[str, Any]]:
    """
    List stored briefs, newest first, without their content.

    :raises RuntimeError: if the database could not be read.
    """
    try:
        return list_analysis_briefs(user_email=user_email, status=status, limit=limit)
    except Exception as e:
        raise RuntimeError(f"Failed to list analysis briefs: {e}") from e
