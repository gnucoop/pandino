import os
import logging
from typing import Optional
from database_pg import get_prompt_from_db


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

    logging.warning(f"Prompt '{title}' not found in DB. Using in-code default_text.")
    if default_text:
        return default_text

    if fallback_env_var:
        logging.warning(f"Default_text missing; trying env var '{fallback_env_var}'")
        return os.getenv(fallback_env_var, "")

    return ""


def render_prompt(template: str, **kwargs) -> str:
    """
    Safely substitute placeholders in a prompt template.

    Example:
        template = "Hello {name}, today is {day}"
        render_prompt(template, name="Gustavo", day="Thursday")

    :param template: The prompt template containing placeholders in {curly braces}.
    :param kwargs: Key-value pairs for substitution.
    :return: The rendered prompt string with placeholders replaced.
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        missing_key = e.args[0]
        logging.warning(
            f"Missing placeholder '{missing_key}' in render_prompt substitution."
        )
        return template
    except Exception as e:
        logging.error(f"Unexpected error in render_prompt: {str(e)}")
        return template
