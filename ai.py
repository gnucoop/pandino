import logging
import os
import re
from dotenv import load_dotenv
from pandasai.llm import BambooLLM
from database_pg import get_user_by_username, log_token_usage
from vector_store import VectorStore
from dataclasses import dataclass
from typing import Optional, Union, List, Any, cast
from pydantic import SecretStr

# Import specific chat models from their respective libraries
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


# Import specific embeddings models from their respective libraries
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import DeepInfraEmbeddings

load_dotenv()  # Load environment variables from .env file


@dataclass
class CompletionRequest:
    username: str
    info: list[str]
    chat: list[str]


@dataclass
class CompletionResponse:
    error: Optional[str] = None
    answer: Optional[str] = None
    vectors: Optional[list[dict]] = None


def choose_llm(
    llm_type: str,
    model: str,
    temperature: float = 0,
    seed: int = 26,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """
    Choose and initialize the appropriate LLM client based on the given type and model.

    :param llm_type: Provider identifier (e.g., 'Groq', 'OpenAI', 'Anthropic').
    :param model: Model name or version string.
    :param temperature: Sampling temperature for the model (default 0).
    :param seed: Random seed for reproducibility if supported by the provider.
    :param base_url: Optional base URL for custom/self-hosted endpoints.
    :param api_key: Optional API key override. Falls back to environment variables if not provided.
    :return: An initialized chat model instance compatible with LangChain.
    :raises ValueError: If the llm_type is unsupported or required environment variables are missing.
    """
    logging.info(f"Choosing LLM: type={llm_type}, model={model}")

    if llm_type == "Groq":
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=SecretStr(os.getenv("GROQ_API_KEY") or ""),
            model_kwargs={"seed": seed},
        )
    elif llm_type == "Deepseek":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            seed=seed,
            base_url="https://api.deepseek.com",
            api_key=SecretStr(os.getenv("DEEPSEEK_API_KEY") or ""),
        )
    elif llm_type == "Deepinfra":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            seed=seed,
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=SecretStr(os.getenv("DEEPINFRA_API_KEY") or ""),
        )
    elif llm_type == "BambooLLM":
        return cast(BaseChatModel, BambooLLM(api_key=os.getenv("PANDASAI_API_KEY")))
    elif llm_type == "Together":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            seed=seed,
            base_url="https://api.together.xyz/v1",
            api_key=SecretStr(os.getenv("TOGETHER_API_KEY") or ""),
        )
    elif llm_type == "Google":
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            seed=seed,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    elif llm_type == "Mistral":
        logging.info("Note: ChatMistralAI does not support seed parameter")
        return ChatMistralAI(
            model_name=model,
            temperature=temperature,
            api_key=SecretStr(os.getenv("MISTRAL_API_KEY") or ""),
        )
    elif llm_type == "Anthropic":
        return ChatAnthropic(
            model_name=model,
            temperature=temperature,
            api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY") or ""),
            model_kwargs={"seed": seed},
            stop=None,
            timeout=None,
        )
    elif llm_type == "OpenAI":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            seed=seed,
            api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),
        )
    elif llm_type == "Ollama":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="http://localhost:11434/v1",
            api_key=SecretStr("ollama" or ""),
        )
    elif llm_type == "Llama.cpp":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="http://localhost:8080/v1",
            api_key=SecretStr("ollama" or ""),
        )
    else:
        logging.error(f"Unsupported llm_type: {llm_type}")
        raise ValueError(f"Unsupported llm_type: {llm_type}")


def choose_emb_model(
    emb_llm_type: str,
    emb_model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Embeddings:
    """
    Choose and initialize the appropriate embeddings model based on the provided type and model.

    :param emb_llm_type: Provider identifier (e.g., 'Mistral', 'Ollama', 'OpenAI').
    :param emb_model: Embedding model name.
    :param base_url: Optional base URL override (used e.g. for Ollama).
    :param api_key: Optional API key override. Falls back to environment variables if not provided.
    :return: Initialized embeddings model instance.
    :raises ValueError: If the provider is unsupported or required configuration is missing.
    """
    logging.info(f"Choosing Embeddings: type={emb_llm_type}, model={emb_model}")

    if emb_llm_type == "Mistral":
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            logging.error("MISTRAL_API_KEY environment variable is not set")
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        return MistralAIEmbeddings(model=emb_model, api_key=SecretStr(key))

    elif emb_llm_type == "OpenAI":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            logging.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return OpenAIEmbeddings(model=emb_model, api_key=SecretStr(key))

    elif emb_llm_type == "Ollama":
        url = base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        return OllamaEmbeddings(model=emb_model, base_url=url)

    elif emb_llm_type == "Deepinfra":
        key = api_key or os.getenv("DEEPINFRA_API_KEY")
        if not key:
            logging.error("DEEPINFRA_API_KEY environment variable is not set")
            raise ValueError("DEEPINFRA_API_KEY environment variable is not set")
        return DeepInfraEmbeddings(model_id=emb_model, deepinfra_api_token=key)

    else:
        logging.error(f"Unsupported emb_llm_type: {emb_llm_type}")
        raise ValueError(f"Unsupported emb_llm_type: {emb_llm_type}")


def complete_chat(
    req: CompletionRequest, store: VectorStore, llm_type: str, model: str
) -> CompletionResponse:
    """
    Perform a chat completion using a vector store for contextual information.

    :param req: CompletionRequest object containing user info and chat history.
    :param store: VectorStore instance used to retrieve relevant context.
    :param llm_type: Provider type (e.g., 'OpenAI', 'Anthropic', etc.).
    :param model: Model name/version to be used for completion.
    :return: CompletionResponse containing the generated answer and optional vectors.
    """

    if not req.chat or not isinstance(req.chat, list):
        return CompletionResponse(error="No chat history provided.")

    question = req.chat[-1].strip()
    if not question:
        return CompletionResponse(error="No question found in chat history.")

    logging.info(f"Starting chat completion with llm_type: {llm_type}, model: {model}")
    logging.info(f"Processing question: {question}")

    vectors: list[dict[str, Any]] = []

    try:
        vectors = store.find_similar_vectors(text=question, top_k=3, min_similarity=0.5)
        logging.info(f"Found {len(vectors)} relevant paragraphs")
    except Exception as e:
        error_msg = f"Vector retrieval failed: {str(e)}"
        logging.error(error_msg)
        return CompletionResponse(error=error_msg)

    if not req.info and not vectors:
        return CompletionResponse(answer="Non ho informazioni al riguardo")

    messages = [
        {
            "role": "system",
            "content": """
            You are Dino, an assistant who helps users by answering questions concisely.
            You will receive information divided by
            BACKGROUND INFORMATION:
            Here you will find the context of previous reply
            RELEVANT CONTENT
            Here you will find context to reply to CURRENT QUESTION
            PREVIUOS CONVERSATION CONTEXT
            you will find here the chat history
            CURRENT QUESTION
            the question that you should reply following the important instruction below

            IMPORTANT INSTRUCTIONS:
            1. You MUST ALWAYS check the provided context and information to answer questions
            2. You MUST ONLY use information from the provided context to answer
            3. You MUST NOT make up or infer information not present in the context
            4. You MUST NEVER say 'I have no information about this' if there is ANY relevant information in the context
            5. If you find ANY relevant information in the context, use it to provide a partial answer
            6. Only say 'I have no information about this' if the context contains ABSOLUTELY NOTHING relevant to the question
            """,
        }
    ]

    # Format context with clear sections and metadata
    context = ""
    # if req.info:
    #    context_parts.append("BACKGROUND INFORMATION:\n-------------------\n" + "\n".join(req.info))
    if vectors:
        context += "RELEVANT CONTEXT:\n----------------"
    for vec in vectors:
        context += "\n" + vec["metadata"]["text"]

    if context:
        messages.append(
            {
                "role": "user",
                "content": "Here is the context you MUST use to answer questions:\n\n"
                + context,
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": "I have received the context and will ONLY use this information to answer questions. I will not make up or infer information not present in this context.",
            }
        )

    # Add the chat history if it exists
    if len(req.chat) > 1:
        messages.append(
            {
                "role": "user",
                "content": "PREVIUOS CONVERSATION CONTEXT:\n-------------------------",
            }
        )
        for i in range(0, len(req.chat) - 1, 2):
            messages.append(
                {"role": "assistant", "content": f"ASSISTANT: {req.chat[i]}"}
            )
            messages.append({"role": "user", "content": f"USER: {req.chat[i+1]}"})

    # Add the final user question with very explicit instructions
    messages.append(
        {
            "role": "user",
            "content": (
                f"CURRENT QUESTION:\n"
                f"----------------\n"
                f"{req.chat[-1]}\n\n"
                "IMPORTANT INSTRUCTIONS:\n"
                "1. Search through ALL the context provided above\n"
                "2. Find ANY relevant information that relates to this question\n"
                "3. If you find ANY relevant information, use it to answer\n"
                "4. Only say 'I have no information about this' if you find ABSOLUTELY NOTHING relevant\n"
                "5. Your answer must ONLY use information from the provided context"
            ),
        }
    )

    try:
        llm = choose_llm(llm_type, model)
        resp = llm.invoke(messages)
        answer = resp.content

        # Extract token usage if available
        token_usage = getattr(resp, "response_metadata", {}).get("token_usage", {})
        token_in = token_usage.get("prompt_tokens", 0)
        token_out = token_usage.get("completion_tokens", 0)

        user = get_user_by_username(req.username)
        if user and (token_in > 0 or token_out > 0):
            log_token_usage(
                user_id=user.get("id"),
                token_input=token_in,
                token_output=token_out,
                model=model,
                provider=llm_type,
            )

        # Ensure answer is string before running .lower()
        answer_text = answer if isinstance(answer, str) else str(answer)

        no_info_phrases = [
            "Non ho informazioni",
            "I have no information",
            "I don't have any information",
            "No information available",
        ]
        is_no_info = any(
            phrase.lower() in answer_text.lower() for phrase in no_info_phrases
        )

        if is_no_info:
            return CompletionResponse(answer=answer_text)
        else:
            return CompletionResponse(answer=answer_text, vectors=vectors)

    except Exception as e:
        logging.exception("Error in chat completion")
        return CompletionResponse(error=f"Chat completion failed: {str(e)}")


def reply_to_prompt(prompt: str, username: str, llm_type: str, model: str) -> str:
    """
    Generate a single text response from a structured prompt, using a fixed system message.

    :param prompt: User-provided prompt content (in Italian).
    :param username: The user requesting the response.
    :param llm_type: LLM provider (e.g. OpenAI, Groq, etc.).
    :param model: Specific model to be used.
    :return: A plain-text response generated by the model.
    :raises Exception: If model invocation fails.
    """
    if not prompt.strip():
        logging.warning("Empty prompt provided to reply_to_prompt")
        return ""

    messages = [
        {
            "role": "system",
            "content": (
                "Sei un esperto di enti non-profit e devi realizzare il rapporto annuale della tua organizzazione.\n"
                "Io ti chiederò di scrivere una sezione alla volta, dandoti indicazioni sui contenuti da includere in ciascuna sezione.\n"
                "Usa un linguaggio preciso ma non troppo tecnico, che sia comprensibile anche al pubblico generale.\n"
                "Non usare elenchi puntati o numerati. Non inserire titoli. Non aggiungere testo all'inizio o alla fine.\n"
                'Non aggiungere paragrafi di conclusione o chiusura. Non usare espressioni come "in questo documento" usa invece "in questa sezione".\n'
                "Scrivi sempre in italiano e genera l'output solo testo senza markdown o html.\n"
                "Se non hai informazioni sufficienti per rispondere non rispondere niente."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        llm = choose_llm(llm_type, model, temperature=0.8)
        resp = llm.invoke(messages)
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    except Exception as e:
        logging.exception(f"Error in prompt completion: {str(e)}")
        raise


def describe_image(url: str, provider: str, model: str) -> str:
    """
    Generates a short description of an image by invoking a vision-capable LLM.

    :param url: Publicly accessible URL of the image to describe.
    :param provider: LLM provider name (e.g., 'OpenAI', 'Google').
    :param model: Specific model identifier that supports image input.
    :return: A short textual description generated by the model.
    :raises Exception: If the model invocation fails.
    """
    logging.info(
        f"Describing image from URL: {url} using provider: {provider}, model: {model}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "descrivi brevemente il contenuto di questa immagine",
                },
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]

    try:
        llm = choose_llm(provider, model, temperature=0.8)
        response = llm.invoke(messages)
        return (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
    except Exception as e:
        logging.exception("Error while describing image")
        raise


def audioFormPromptBuild(
    formSchemaExampleData: dict[str, Any],
    formSchemaName: str,
    formSchemaChoices: list[dict[str, Any]],
    transcribedAudio: str,
) -> dict[str, str]:
    """
    Builds a pair of prompts (system and user) to instruct an LLM to populate a JSON form
    based on a given schema, example data, available choices, and a transcribed audio input.

    :param formSchemaExampleData: Dictionary with an example of the form compiled with empty/default values.
    :param formSchemaName: Name of the form, used in the prompts.
    :param formSchemaChoices: List of dictionaries representing selectable choices for choice-based fields.
    :param transcribedAudio: Transcribed user audio input, from which field values will be extracted.
    :return: A dictionary with 'systemprompt' and 'userprompt' keys, both containing formatted strings.
    :raises ValueError: If any required input is missing.
    """
    if not formSchemaExampleData or not formSchemaName or not transcribedAudio:
        raise ValueError(
            "Missing one or more required inputs for building audio form prompts"
        )

    fieldTypes = formSchemaExampleData["fieldTypes"]
    fieldDescriptions = formSchemaExampleData["fieldDescriptions"]

    logging.info("Building audio form prompts for schema: %s", formSchemaName)

    system = f"""
    Sei un assistente specializzato nell'estrazione di dati da trascrizioni audio.
    Rispondi ESCLUSIVAMENTE in formato JSON valido.
    Non aggiungere commenti, spiegazioni o testo aggiuntivo.
    """

    user = f"""
    DATI INPUT:
    Nome dello schema del form: {formSchemaName}
    Opzioni disponibili: {formSchemaChoices}
    Template di output e tipi dei campi: {fieldTypes}
    Descrizione dei campi: {fieldDescriptions}
    Trascrizione audio: {transcribedAudio}
    
    ISTRUZIONI:
    Compila il template JSON utilizzando SOLO le informazioni dalla trascrizione.
    REGOLE PER CAMPO:
    - boolean: true/false basato sulla trascrizione
    - multiplechoice: array di valori da "Opzioni disponibili". Se menzionata opzione non presente e se tra le Opzioni disponibili esiste "altro", includi "altro"
    - singlechoice: array di valori da "Opzioni disponibili". Se menzionata opzione non presente e se tra le Opzioni disponibili esiste "altro", includi "altro"
    - date: formato YYYY-MM-DD
    - text/string: testo estratto dalla trascrizione
    - range/number: valore numerico
    OUTPUT: JSON compilato seguendo il template fornito.
    """

    return {"systemprompt": system.strip(), "userprompt": user.strip()}


def audioFormCompilation(
    userprompt: str, systemprompt: str, username: str, llm_type: str, model: str
) -> Union[str, CompletionResponse]:
    """
    Sends a user/system prompt pair to a selected LLM and returns the generated content.

    :param userprompt: Prompt containing audio-derived user input instructions.
    :param systemprompt: Prompt defining rules and expected structure of response.
    :param username: User requesting the form completion.
    :param llm_type: Type/provider of the language model (e.g., 'OpenAI', 'Anthropic').
    :param model: Specific model name to invoke.
    :return: A string (LLM response content) on success, or a CompletionResponse on failure.
    """
    if not all([userprompt, systemprompt, llm_type, model, username]):
        raise ValueError(
            "Missing one or more required parameters for audioFormCompilation"
        )

    logging.info(
        f"Invoking audio form compilation with model={model} (provider={llm_type}) for user={username}"
    )

    messages = [
        {"role": "system", "content": systemprompt},
        {"role": "user", "content": userprompt},
    ]

    llm = choose_llm(llm_type, model, temperature=0)

    try:
        resp = llm.invoke(messages)

        token_usage = getattr(resp, "response_metadata", {}).get("token_usage", {})
        token_in = token_usage.get("prompt_tokens", 0)
        token_out = token_usage.get("completion_tokens", 0)

        user = get_user_by_username(username)
        if user and (token_in > 0 or token_out > 0):
            log_token_usage(
                user_id=user.get("id"),
                token_input=token_in,
                token_output=token_out,
                model=model,
                provider=llm_type,
            )

        return resp.content if isinstance(resp.content, str) else str(resp.content)

    except Exception as e:
        logging.exception("Error in audio form compilation")
        return CompletionResponse(error=f"Audio form compilation failed: {str(e)}")
