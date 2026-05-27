import textwrap
import time
from typing import TypedDict

from smolagents import CodeAgent

from config import AppConfig
from llm.litellm_factory import build_litellm_model
from retriever_tool import RetrieverTool
from prompt_utils import load_prompt, render_prompt
from utils.agent_serialization import serialize_runresult
from utils.agent_logging import log_runresult


DEFAULT_AGENTCHAT_PROMPT = textwrap.dedent("""\
    You are "Compass AI Tutor", an assistant embedded in the Compass training platform.

    PURPOSE
    - Answer user questions about topics available in the selected namespace: "{namespace}".
    - Use ONLY the information retrieved via the `retriever` tool. Do NOT rely on your general pre-trained knowledge.
    - Always call the retriever BEFORE answering. If the first retrieval is insufficient, try again with semantically different queries.

    INPUTS
    - User question: "{user_question}"

    INSTRUCTIONS
    1) Read all retrieved context passages and synthesize a clear, technically-accurate answer in the language indicated by the variable {language}.
    2) Maintain a cordial but neutral and technical tone (no hype, no speculation).
    3) After the answer, produce 2–3 suggested follow-up questions (in the language indicated by the variable {language}) that the user might ask to deepen understanding.
    4) OUTPUT FORMAT: return a VALID JSON object with exactly these top-level fields:
    - "answer": string
    - "follow_ups": array of strings
    5) If no context passages are retrieved or they are empty, return a valid JSON object with the following structure:
    - "answer": "Mi dispiace, non ho trovato informazioni sufficienti nel materiale disponibile per rispondere con precisione.",
    - "follow_ups": []

    LANGUAGE
    - Always respond in the language indicated by the variable {language}.
""")


class AgentChatServiceResult(TypedDict):
    payload: dict
    model: str
    provider: str


def run_agentchat(
    chat: list,
    namespace: str,
    language: str,
    username: str,
    config: AppConfig,
) -> AgentChatServiceResult:
    try:
        provider = config.models.completion_model_provider
        configured_model = config.models.completion_model_agent_chat

        if not configured_model:
            raise RuntimeError("COMPLETION_MODEL_AGENT_CHAT is not configured.")

        model_clean = configured_model

        llm = build_litellm_model(
            provider=provider,
            configured_model=configured_model,
            temperature=0,
        )

        retriever_tool = RetrieverTool(
            namespace=namespace,
            embedding_provider=config.models.completion_embedding_model_provider,
            embedding_model=config.models.completion_embedding_model,
            top_k=config.rag.top_k,
            min_sim=config.rag.min_sim,
        )

        agent = CodeAgent(
            tools=[retriever_tool],
            model=llm,
            max_steps=5,
            additional_authorized_imports=["json"],
        )

        user_message = chat[-1]

        base_prompt_template = load_prompt(
            "compass_agentchat_system", default_text=DEFAULT_AGENTCHAT_PROMPT
        )

        system_prompt = render_prompt(
            base_prompt_template,
            namespace=namespace,
            user_question=user_message,
            language=language,
        )

        start_time = time.time()

        result = agent.run(
            user_message,
            additional_args={"system_prompt": system_prompt},
            return_full_result=True,
        )

        duration_ms = round((time.time() - start_time) * 1000, 2)

        payload = serialize_runresult(result)

        no_answer_fallback = (
            "Mi dispiace, non ho trovato informazioni sufficienti nel "
            "materiale disponibile per rispondere con precisione."
        )

        if not str(payload.get("answer", "")).strip():
            payload["answer"] = no_answer_fallback

        payload["metrics"]["duration_ms"] = duration_ms

        log_runresult(
            result,
            user=username,
            namespace=namespace,
            language=language,
            question=user_message,
        )

        return AgentChatServiceResult(
            payload=payload,
            model=model_clean,
            provider=provider,
        )

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"agentchat_service failed: {e}") from e
