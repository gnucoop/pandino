from config import InterviewerConfig
from interviewer.engine import InterviewerEngine


def create_interviewer_engine(
    *,
    api_key: str,
    user_name: str,
    config: InterviewerConfig,
    lang: str = "ENG",
) -> InterviewerEngine:
    return InterviewerEngine(
        api_key=api_key,
        user_name=user_name,
        config=config,
        lang=lang,
    )
