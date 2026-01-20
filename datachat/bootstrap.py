
import textwrap
import pandas as pd
from prompt_utils import load_prompt, render_prompt


def build_language_instruction(lang: str) -> str:
    return (
        "Please answer using the official language of the country "
        "corresponding to the following ISO 3166-1 alpha-3 code: "
        f"{lang}. "
        "If you can't match the language, please answer in English."
    )


def build_start_prompt(data: pd.DataFrame) -> str:
    default_startchat_prompt = textwrap.dedent("""\
        This is a pandas dataframe: {data}
        Try to understand the nature of the data and suggest me what kind of analysis should I ask for.
        Explain in details your answers and make any suggestions about possible questions that I could ask.
        Do not suggest any python code.
        Please reply in a readable HTML format, with no asterisks and adding a line break after each paragraph.
    """)

    base_prompt_template = load_prompt(
        "start_chat_system",
        default_text=default_startchat_prompt,
    )

    return render_prompt(base_prompt_template, data=data)


def build_bootstrap_question(data: pd.DataFrame, lang: str) -> str:
    language_instruction = build_language_instruction(lang)
    base_prompt = build_start_prompt(data)
    return f"{language_instruction}\n\n{base_prompt}"
