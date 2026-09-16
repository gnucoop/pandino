import os
import logging
from typing import Optional
from infrastructure.database_pg import get_prompt_from_db

logger = logging.getLogger(__name__)


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

    Only the placeholders named in `kwargs` are substituted. Every other brace in
    the template is literal and is passed through untouched, so a prompt may
    contain JSON examples such as {"kind":"text"} without being mangled.

    Not supported, because no prompt uses them: format specs and conversions
    ({x:>10}, {x!r}). Those are treated as literal text.

    :param template: The prompt template containing placeholders in {curly braces}.
    :param kwargs: Key-value pairs for substitution.
    :return: The rendered prompt string with placeholders replaced.
    """
    # Escape everything, then re-open only the placeholders we were given a
    # value for. What is left escaped is literal and survives .format().
    escaped = template.replace("{", "{{").replace("}", "}}")
    for key in kwargs:
        escaped = escaped.replace("{{" + key + "}}", "{" + key + "}")

    try:
        return escaped.format(**kwargs)
    except KeyError as e:
        missing_key = e.args[0]
        logger.warning(
            "event=prompt_placeholder_missing key=%s", missing_key
        )
        return template
    except Exception as e:
        logger.error("event=prompt_render_failed error=%s", str(e))
        return template
