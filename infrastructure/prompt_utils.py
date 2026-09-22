import logging
import os
import re
from typing import Optional
from infrastructure.database_pg import get_prompt_from_db

logger = logging.getLogger(__name__)

#: A placeholder is a bare Python identifier in single braces - and nothing
#: else. Anchoring on the identifier is what keeps JSON object syntax such as
#: {"kind":"text"} out of the match set.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_prompt(
    title: str, 
    default_text: str = "", 
    fallback_env_var: Optional[str] = None,
    version: Optional[int] = None) -> str:
    """
    Retrieve a prompt message using a multi-step fallback strategy.

    Order of precedence:
    1. Database (table 'prompts', field 'message')
    2. Default text provided in code
    3. Environment variable (if specified)
    4. Empty string if all fail
    """
    prompt = get_prompt_from_db(title, version=version)
    if prompt:
        return prompt

    logger.warning("event=prompt_default_used title=%s", title)
    if default_text:
        return default_text

    if fallback_env_var:
        logger.warning("event=prompt_env_fallback_used env_var=%s", fallback_env_var)
        return os.getenv(fallback_env_var, "")

    return ""


def render_prompt(template: str, **kwargs) -> str:
    """
    Safely substitute placeholders in a prompt template.

    Example:
        template = "Hello {name}, today is {day}"
        render_prompt(template, name="Gustavo", day="Thursday")

    A *placeholder* is defined strictly: a single pair of braces around a bare
    Python identifier, ``{name}``. Nothing else is one. That is what lets a
    prompt carry JSON examples such as ``{"kind":"text"}`` - the brace is
    followed by a quote, not an identifier, so it is never a candidate and is
    passed through untouched.

    Substitution is a single regex pass rather than ``str.format``, so:

    - every brace that is not a strict placeholder stays literal, including
      ``{{`` (no format-style unescaping happens, because no format happens);
    - a recognizable placeholder the caller did not supply is left literal in
      the output *and* reported via ``event=prompt_placeholder_missing``. The
      old all-or-nothing behaviour - discard the whole render, return the raw
      template - is deliberately not restored: one stale placeholder in a
      DB-stored prompt must not cost the caller every other substitution.

    Not supported, because no prompt uses them: format specs and conversions
    ({x:>10}, {x!r}). Those do not match the strict form and stay literal.

    :param template: The prompt template containing placeholders in {curly braces}.
    :param kwargs: Key-value pairs for substitution. Unused keys are ignored.
    :return: The rendered prompt string with supplied placeholders replaced.
    """
    missing: list[str] = []

    def _substitute(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key in kwargs:
            return str(kwargs[key])
        missing.append(key)
        return match.group(0)

    try:
        rendered = _PLACEHOLDER_RE.sub(_substitute, template)
    except Exception as e:
        # Reachable: _substitute calls str() on caller-supplied values, and a
        # value whose __str__ raises propagates out of re.sub. Rare, but a
        # render that blows up must not take the caller down with it.
        logger.error("event=prompt_render_failed error=%s", str(e))
        return template

    for key in dict.fromkeys(missing):
        logger.warning("event=prompt_placeholder_missing key=%s", key)

    return rendered
