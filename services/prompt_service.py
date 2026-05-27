import logging
from typing import TypedDict

from infrastructure.ai import choose_llm
from infrastructure.prompt_utils import load_prompt, render_prompt


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int


class PromptServiceResult(TypedDict):
    content: str
    token_usage: TokenUsage


def reply_to_prompt(
    prompt: str,
    llm_type: str,
    model: str,
    language: str = "ITA",
    api_key: str | None = None,
) -> PromptServiceResult:
    if not prompt.strip():
        logging.warning("Empty prompt provided to reply_to_prompt")
        return PromptServiceResult(
            content="", token_usage=TokenUsage(input_tokens=0, output_tokens=0)
        )

    language_instruction = (
        f"Please answer using the official language of the country corresponding to the following ISO 3166-1 alpha-3 code: {language}. "
        f"If you can't match the language, please answer in English."
    )

    default_reply_to_prompt = (
        "You are an expert in non-profit organizations and you have to create the annual report for your organization.\n"
        "I will ask you to write one section at a time, giving you instructions on the content to include in each section.\n"
        "Use precise but not overly technical language that is understandable to the general public.\n"
        "Do not use bulleted or numbered lists. Do not insert titles. Do not add text at the beginning or at the end.\n"
        "Do not add concluding or closing paragraphs. Do not use expressions like 'in this document'; use 'in this section' instead.\n"
        "Always write in the language specified by the language instruction and generate the output as plain text without markdown or html.\n"
        "If you do not have enough information to answer, do not answer anything."
    )

    base_prompt_template = load_prompt(
        "reply_to_prompt_system", default_text=default_reply_to_prompt
    )

    base_prompt = render_prompt(base_prompt_template)

    full_prompt = f"{language_instruction}\n\n{base_prompt}"

    messages = [
        {"role": "system", "content": full_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        llm = choose_llm(llm_type, model, temperature=0.8, api_key=api_key)
        resp = llm.invoke(messages)
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        input_tokens = getattr(resp.usage_metadata, "input_tokens", 0) or 0
        output_tokens = getattr(resp.usage_metadata, "output_tokens", 0) or 0
        return PromptServiceResult(
            content=content,
            token_usage=TokenUsage(
                input_tokens=input_tokens, output_tokens=output_tokens
            ),
        )
    except Exception as e:
        raise RuntimeError(f"Error in reply_to_prompt: {e}") from e
