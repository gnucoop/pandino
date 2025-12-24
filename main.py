# === Built-in ===
import os
import base64
import math
import secrets
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from typing import Any, List, Union, Optional
import textwrap
import logging
import time
from functools import wraps
import io

# === Third-party ===
from flask import (
    Flask, 
    request, 
    Response, 
    jsonify, 
    abort, 
    render_template,
    redirect, 
    url_for, 
    session, 
    flash
)
from flask_cors import CORS
import pandas as pd
from pandasai import Agent
from smolagents import CodeAgent
from smolagents.models import LiteLLMModel
import matplotlib
import requests
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
import bcrypt
import psutil

# === Local modules ===
from agent_manager import getAgent, createAgent, deleteAgent
from file_manager import isImageFilePath, fileToBase64
from retriever_tool import RetrieverTool
from vector_store import PineconeStore, PGVectorStore, merge_segments
import database_pg
from database_pg import (
    edit_tokens, 
    validate_api_key, 
    get_users_for_admin, 
    get_users_stats, 
    get_logs_for_admin, 
    get_logs_stats, 
    update_user_tokens, 
    get_user_by_id, 
    get_all_prompts, 
    get_prompt_by_id, 
    add_prompt, 
    update_prompt, 
    delete_prompt, 
    get_all_costs, 
    add_cost, 
    update_cost, 
    delete_cost, 
    get_cost_by_id, 
    get_daily_stats, 
    get_recent_activity, 
    log_token_usage,
    get_feedback_for_admin,
    get_feedback_stats
)

from dino import dino_authenticate
from ai import (
    audioFormCompilation,
    audioFormPromptBuild,
    CompletionResponse,
    CompletionRequest,
    describe_image,
    complete_chat,
    reply_to_prompt,
    choose_llm,
    choose_emb_model
)
from prompt_utils import load_prompt, render_prompt
from utils.agent_serialization import serialize_runresult
from utils.agent_logging import log_runresult, setup_agent_logger
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

# Initialize the Flask application
app = Flask(__name__)
# origins=["http://localhost:4200"]
CORS(app)

secret_key = os.environ.get('ENCRYPTION_KEY')
if not secret_key:
    raise RuntimeError("ENCRYPTION_KEY must be set in environment")
app.secret_key = secret_key

# Configure the agent run logger
setup_agent_logger()

# Verify Matplotlib backend
print(f"Matplotlib backend: {matplotlib.get_backend()}")

# Removing Pandas read csv columns limitations to avoid truncated dataFrames
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

DATACHAT_MODEL = os.environ.get("DATACHAT_MODEL")
DATACHAT_PROVIDER = os.environ.get("DATACHAT_PROVIDER")
PROMPT_MODEL = os.environ.get("PROMPT_MODEL")
PROMPT_PROVIDER = os.environ.get("PROMPT_PROVIDER")
AUDIO_MODEL = os.environ.get("AUDIO_MODEL")
AUDIO_PROVIDER = os.environ.get("AUDIO_PROVIDER")
COMPLETION_MODEL = os.environ.get("COMPLETION_MODEL")
COMPLETION_MODEL_PROVIDER = os.environ.get("COMPLETION_MODEL_PROVIDER")
COMPLETION_MODEL_AGENT_CHAT = os.environ.get("COMPLETION_MODEL_AGENT_CHAT")
COMPLETION_EMBEDDING_MODEL = os.environ.get("COMPLETION_EMBEDDING_MODEL")
COMPLETION_EMBEDDING_MODEL_PROVIDER = os.environ.get(
    "COMPLETION_EMBEDDING_MODEL_PROVIDER"
)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL")
VISION_PROVIDER = os.environ.get("VISION_PROVIDER")
VISION_MODEL = os.environ.get("VISION_MODEL")
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
STRIPE_KEY = os.environ.get("STRIPE_SK_KEY")
DATACHAT_TOKEN_COST = int(os.environ.get("DATACHAT_TOKEN_COST", "1"))
COMPLETION_TOKEN_COST = os.environ.get("COMPLETION_TOKEN_COST")
PROMPT_TOKEN_COST = os.environ.get("PROMPT_TOKEN_COST")
AUDIO_FORM_TOKEN_COST = os.environ.get("AUDIO_FORM_TOKEN_COST")
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH', '').encode("utf-8")


# Admin authentication decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('Please log in to access the admin panel', 'warning')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function


# Define a route for the '/' endpoint that returns a welcome message
@app.route("/")
def welcome() -> str:
    return "Welcome to Pandino! This is the root endpoint."


# Validates an API Key associated to an user email
def assert_valid_api_key(api_key: str, user_email: str) -> None:
    """
    Validate the provided API key for the given user email and abort the request if invalid.

    :param api_key: API key string to be validated.
    :param user_email: Email address of the user associated with the API key.
    :return: None
    :raises werkzeug.exceptions.HTTPException: Aborts with 403 if the API key is missing, expired, or invalid.
    """
    if not api_key:
        abort(403, description="Missing API key")
    result, message = validate_api_key(api_key, user_email)
    if not result:
        if "expired" in message:
            abort(403, description="API key expired")
        else:
            abort(403, description="Invalid API key")


# Recursively replace NaN with None in dictionaries or lists.
def replace_nan(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: replace_nan(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan(item) for item in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    else:
        return data


# Define a route for the '/edittokens' endpoint that accepts POST requests
@app.route("/edittokens", methods=["POST"])
def editTokens() -> tuple[Response, int]:
    try:
        stripe_key = request.headers.get("X-STRIPE-KEY")

        # Check stripe_key is present and correct
        if not stripe_key:
            return jsonify({"error": "Missing X-STRIPE-KEY header"}), 400

        if stripe_key != STRIPE_KEY:
            return jsonify({"error": "Invalid STRIPE KEY"}), 403

        r = request.get_json()
        if not r:
            return jsonify({"error": "No JSON data provided"}), 400

        required_keys = ["quantity", "useremail"]
        missing_keys = [key for key in required_keys if key not in r]

        if missing_keys:
            return (
                jsonify({"error": f"Missing required keys: {', '.join(missing_keys)}"}),
                400,
            )

        result, message = edit_tokens(r["useremail"], r["quantity"])

        if result:
            return (
                jsonify(
                    {
                        "response": f"{message}: {r['quantity']} for user: {r['useremail']}"
                    }
                ),
                200,
            )
        elif not result:
            return (
                jsonify({"error": f"{message}"}),
                400,
            )

    except Exception as e:
        app.logger.error(f"Unexpected error in edit tokens: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

    return jsonify({"error": "Unhandled case in editTokens"}), 500


# Define a route for the '/edittokens' endpoint that accepts POST requests
@app.route("/getusertokens", methods=["POST"])
def getUserTokens() -> tuple[Response, int]:
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400
    assert_valid_api_key(api_key, user_email)
    tokens = database_pg.get_user_tokens(user_email)
    return jsonify({"response": {"tokens": tokens}}), 200


# Define a route for the '/adduser' endpoint that accepts POST requests
@app.route("/checkpandinouser", methods=["POST"])
def addNewUser() -> Union[tuple[Response, int], tuple[str, int, dict[str, str]]]:
    graphql_url = request.headers.get("X-GRAPHQL-URL")
    auth_token = request.headers.get("X-AUTH-TOKEN")
    user_email = request.headers.get("X-USER-EMAIL")
    if not graphql_url:
        return jsonify({"error": "Missing X-GRAPHQL-URL header"}), 400
    if not auth_token:
        return jsonify({"error": "Missing X-AUTH-TOKEN header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    err = dino_authenticate(graphql_url, auth_token)
    if err:
        return str(err), 403, {"Content-Type": "text/plain"}

    existingUser = database_pg.get_user_by_username(user_email)
    if not existingUser:
        generatedKey = secrets.token_urlsafe(8)
        currentDate = datetime.now()
        expirationDate = currentDate.replace(year=currentDate.year + 2)
        addUserResult = database_pg.add_user(
            user_email, generatedKey, expirationDate.strftime("%Y-%m-%d %H:%M:%S")
        )
        if addUserResult is None:
            return (
                jsonify(
                    {
                        "response": {
                            "user": {
                                "user_email": user_email,
                                "api_key": generatedKey,
                                "expiration_date": expirationDate,
                            }
                        }
                    }
                ),
                200,
            )
        else:
            return (
                jsonify({"error": addUserResult}),
                500,
            )
    else:
        print(existingUser)
        return (
            jsonify(
                {
                    "response": {
                        "user": {
                            "user_email": existingUser.get("user"),
                            "api_key": existingUser.get("api_key"),
                            "expiration_date": existingUser.get("date_valid_until"),
                        }
                    }
                }
            ),
            200,
        )


# Define a route for the '/validateapikey' endpoint that accepts POST requests
@app.route("/validateapikey", methods=["POST"])
def validate() -> tuple[Response, int]:
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    # Check if all required parameters are present
    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400
    result, message = validate_api_key(api_key, user_email)

    if not result:
        if "expired" in message:
            return jsonify({"error": "API key expired"}), 403
        else:
            return jsonify({"error": "Invalid API key"}), 403
    else:
        return jsonify({"response": "API key match found"}), 200


@app.route("/feedback", methods=["POST"])
def feedback_handler() -> Response | tuple[Response, int]:
    """
    Endpoint to collect user feedback on LLM-generated responses.

    This endpoint allows authenticated users to submit a positive or negative
    evaluation of a generated answer, optionally linking it to an existing log
    entry for future analysis. Feedback is stored persistently and does not
    interfere with token usage or response generation flows.
    """
    try:
        # === INPUT VALIDATION ===

        r = request.get_json()
        if not r:
            return jsonify({"error": "No JSON data provided"}), 400

        required = ["username", "question", "answer", "feedback"]
        missing = [k for k in required if k not in r]
        if missing:
            return jsonify(
                {"error": f"Missing required keys: {', '.join(missing)}"}
            ), 400

        api_key = request.headers.get("X-API-KEY")
        if not api_key:
            return jsonify({"error": "Missing X-API-KEY header"}), 400

        # === AUTHENTICATION ===

        assert_valid_api_key(api_key, r["username"])

        # === SEMANTIC VALIDATION ===

        question = r["question"]
        answer = r["answer"]
        feedback_value = r["feedback"]

        if not isinstance(question, str) or not question.strip():
            return jsonify({"error": "Invalid 'question'"}), 400

        if not isinstance(answer, str) or not answer.strip():
            return jsonify({"error": "Invalid 'answer'"}), 400

        if feedback_value not in ("positive", "negative"):
            return jsonify({"error": "Invalid feedback value"}), 400

        log_id = r.get("log_id")
        if log_id is not None and not isinstance(log_id, int):
            return jsonify({"error": "Invalid 'log_id': expected integer"}), 400

        if log_id is not None:
            with database_pg.connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1 FROM logs WHERE id = %s", (log_id,))
                exists = cur.fetchone()

            if not exists:
                return jsonify({"error": f"log_id {log_id} does not exist"}), 400

        source = r.get("source")
        if source is not None and not isinstance(source, str):
            return jsonify({"error": "Invalid 'source'"}), 400

        # === PERSISTENCE ===

        feedback_id = database_pg.save_feedback(
            user_email=r["username"],
            question=question,
            answer=answer,
            feedback_value=feedback_value,
            log_id=log_id,
            source=source,
        )

        # === RESPONSE ===

        return jsonify({"feedback_id": feedback_id}), 200

    except Exception as e:
        app.logger.error(f"[feedback] Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


# Define a route for the '/endchat' endpoint that accepts POST requests
@app.route("/enddatachat", methods=["POST"])
def endChat() -> Response | tuple[Response, int]:

    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    user_name_header = request.headers.get("X-USER-NAME")
    user_name = (
        user_name_header.replace(" ", "_").strip() if user_name_header != None else None
    )

    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400

    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    assert_valid_api_key(api_key, user_email)

    # Check if all required parameters are present
    if not user_name:
        return jsonify({"error": "Missing X-USER-NAME header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    deletedAgent = deleteAgent(api_key, user_name)
    if deletedAgent != None and deletedAgent.conversation_id:
        return jsonify({"Agent deleted succesfully": deletedAgent.conversation_id})
    else:
        return jsonify({"Agent was not active for this key": api_key})


# Define a route for the '/startchat' endpoint that accepts POST requests
@app.route("/startdatachat", methods=["POST"])
def startChat() -> Response | tuple[Response, int]:
    api_key = request.headers.get("X-API-KEY")
    user_name_header = request.headers.get("X-USER-NAME")
    user_email = request.headers.get("X-USER-EMAIL")
    user_name = (
        user_name_header.replace(" ", "_").strip()
        if user_name_header is not None
        else None
    )

    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400

    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    assert_valid_api_key(api_key, user_email)

    # Extract necessary parameters from the request FORMDATA
    request_model_name = request.form.get("model_name")
    request_llm_type = request.form.get("llm_type")
    request_file = request.files.get("file")
    request_lang = request.form.get("lang")
    model_name = request_model_name if request_model_name else DATACHAT_MODEL
    llm_type = request_llm_type if request_llm_type else DATACHAT_PROVIDER
    lang = request_lang if request_lang else "ENG"
    # Check if all required parameters are present
    if (
        not model_name
        or not llm_type
        or not user_name
        or not user_email
        or not request_file
    ):
        return jsonify({"error": "Missing parameters"}), 400

    # Checks if the User's tokens are enough for this operation
    user_tokens = database_pg.get_user_tokens(user_email)

    if user_tokens is None:
        return jsonify({"error": "Could not retrieve user tokens"}), 500

    if int(DATACHAT_TOKEN_COST) > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    # Read the data from the provided CSV file
    data = pd.read_csv(io.StringIO(request_file.stream.read().decode('utf-8')), sep=",")
    # Initialize the language model based on the provided type
    llm = choose_llm(llm_type, model_name)
    # Initialize the agent with the data and configuration
    try:
        agent = createAgent(api_key, data, llm, user_name)

        if agent is None:
            return jsonify({"error": "Agent creation failed"}), 500

        agentResponse: dict[str, Any] = {"Agent active": str(agent.conversation_id)}

        # Language-aware prompt generation
        language_instruction = (
            f"Please answer using the official language of the country corresponding to the following ISO 3166-1 alpha-3 code: {lang}. "
            f"If you can't match the language, please answer in English."
        )
        
        default_startchat_prompt = textwrap.dedent("""\
            This is a pandas dataframe: {data}
            Try to understand the nature of the data and suggest me what kind of analysis should I ask for.
            Explain in details your answers and make any suggestions about possible questions that I could ask.
            Do not suggest any python code.
            Please reply in a readable HTML format, with no asterisks and adding a line break after each paragraph.
        """)
        
        base_prompt_template = load_prompt("start_chat_system", default_text=default_startchat_prompt)
        base_prompt = render_prompt(base_prompt_template, data=data)
        
        question = f"{language_instruction}\n\n{base_prompt}"
        
        logging.info(f"Invoking startdatachat agent with language={lang}, user={user_email}")
        
        suggestionsResponse = llm.invoke(question)
        if suggestionsResponse and suggestionsResponse.content is not None:
            agentResponse.update({"suggested_questions": suggestionsResponse.content})

        # Spends User's tokens
        edit_tokens(user_email, -int(DATACHAT_TOKEN_COST))

        return jsonify(agentResponse)
    except Exception as e:
        return (
            jsonify({"error": f"Failed to create Agent: {str(e)}"}),
            500,
        )


# Define a route for the /datachat endpoint
@app.route("/datachat", methods=["POST"])
def dataChat() -> Response | tuple[Response, int]:

    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")

    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    assert_valid_api_key(api_key, user_email)

    if not request.json:
        return jsonify({"error": "Missing JSON body"}), 400

    chat = request.json.get("chat")
    agent: Agent | None = getAgent(api_key)

    # Check if the Chat parameter is present
    if not chat:
        return jsonify({"error": "Missing Chat string"}), 400

    # Check if user email is present
    if not user_email:
        return jsonify({"error": "Missing User email"}), 400

    # Check if the Agent is active
    if not agent:
        return jsonify({"error": "Agent not active for this Api Key"}), 400

    # Checks if the User's tokens are enough for this operation

    user_tokens = database_pg.get_user_tokens(user_email)

    if user_tokens is None:
        return jsonify({"error": "Could not retrieve user tokens"}), 500

    if int(DATACHAT_TOKEN_COST) > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    # Perform the chat operation and get the response and explanation
    response = agent.chat(chat)
    # explanation = agent.explain()

    # Convert the response to a DataFrame if it's a list
    if isinstance(response, list):
        try:
            response = pd.DataFrame(response)
        except Exception as e:
            return (
                jsonify({"error": f"Failed to convert list to DataFrame: {str(e)}"}),
                500,
            )

    # Convert the response to a dictionary
    if isinstance(response, pd.DataFrame):
        response_dict = {
            "type": "dataframe",
            "value": replace_nan(response.to_dict(orient="records")),
        }
    elif isinstance(response, dict):
        response_dict = replace_nan(response)
        response_dict.update({"type": "dict"})
    else:
        response_dict = {"type": type(response).__name__, "value": str(response)}
        if response_dict and response_dict["value"]:
            # Handle string type response with plot
            if response_dict["type"] == "string" and "plot" in response_dict:
                plot_path = response_dict.get("plot")
                if plot_path and os.path.exists(plot_path):
                    response_dict["type"] = "text_and_image"
                    response_dict["image"] = fileToBase64(plot_path)
                    # Remove the plot path from the response
                    del response_dict["plot"]
            # Convert image file path in value to a base64 serialized file
            elif isImageFilePath(response_dict["value"]):
                response_dict["type"] = "image"
                response_dict["value"] = fileToBase64(response_dict["value"])

    # Spends User's tokens
    edit_tokens(user_email, -int(DATACHAT_TOKEN_COST))

    return jsonify({"response": response_dict, "explanation": None})


# Define a route for the /datachat endpoint
@app.route("/buyreport", methods=["POST"])
def buyReport() -> tuple[Response, int]:

    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    assert_valid_api_key(api_key, user_email)

    if not request.json:
        return jsonify({"error": "Missing JSON body"}), 400
    prompts = request.json.get("prompts")
    if not isinstance(prompts, int):
        return jsonify({"error": "Missing Prompts numeric parameter"}), 400

    user_tokens = database_pg.get_user_tokens(user_email)
    if user_tokens is None:
        return jsonify({"error": "Could not retrieve user tokens"}), 500
    if prompts > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    result, message = edit_tokens(user_email, -prompts)

    return jsonify({"response": result, "message": f"{message}"}), 200


@app.route("/completion.json", methods=["POST"])
def completion_handler() -> Union[Response, tuple[Response, int]]:
    try:
        r = request.get_json()
        if not r:
            return jsonify({"error": "No JSON data provided"}), 400

        required_keys = ["chat", "username"]
        missing_keys = [key for key in required_keys if key not in r]
        if missing_keys:
            return (
                jsonify({"error": f"Missing required keys: {', '.join(missing_keys)}"}),
                400,
            )

        api_key = request.headers.get("X-API-KEY")
        if not api_key:
            return jsonify({"error": "Missing X-API-KEY header"}), 400

        assert_valid_api_key(api_key, r["username"])

        # Token check
        user_tokens = database_pg.get_user_tokens(r["username"])
        if user_tokens is None:
            return jsonify({"error": "Could not retrieve user tokens"}), 500

        token_cost = int(COMPLETION_TOKEN_COST or "1")
        if token_cost > user_tokens:
            return (
                jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}),
                500,
            )

        # Request assembly
        chat_request = CompletionRequest(
            username=r["username"],
            info=r.get("info", []),
            chat=r["chat"],
        )
        namespace = r.get("namespace", "")

        # Scelta modelli
        llm_type = COMPLETION_MODEL_PROVIDER or "google"
        model = COMPLETION_MODEL or "gemini-2.5-flash"
        emb_llm_type = COMPLETION_EMBEDDING_MODEL_PROVIDER or "Deepinfra"
        emb_model = COMPLETION_EMBEDDING_MODEL or "intfloat/multilingual-e5-large-instruct"

        embeddings = choose_emb_model(emb_llm_type, emb_model)

        store = PGVectorStore(embeddings, namespace)
        resp = complete_chat(chat_request, store, llm_type, model)
        if resp and hasattr(resp, "vectors") and resp.vectors:
            for vec in resp.vectors:
                vec["similarity"] += 0.3

        if isinstance(resp, CompletionResponse):
            if resp.error:
                return jsonify({"error": f"Chat completion error: {resp.error}"}), 200

            # Token deduction
            if resp.answer or resp.vectors:
                edit_tokens(r["username"], -token_cost)

            response_payload = {
                "answer": resp.answer,
                "vectors": resp.vectors,
            }

            if resp.log_id is not None:
                response_payload["log_id"] = resp.log_id

            return jsonify(response_payload), 200

        elif resp is None:
            return jsonify({"error": "No response from chat completion"}), 500
        else:
            return jsonify({"error": "Unexpected response format"}), 500

    except Exception as e:
        app.logger.error(f"Unexpected error in completion_handler: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

textContentType = {"Content-Type": "text/plain"}


@app.route("/agentchat", methods=["POST"])
def agentchat() -> Response | tuple[Response, int]:
    """
    AI endpoint based on Smolagents.
    The agent must always use DinoRetrieverTool to retrieve context
    before generating the response.
    """
    try:
        # === INPUT VALIDATION ===
        
        r = request.get_json()
        if not r:
            return jsonify({"error": "No JSON data provided"}), 400

        required = ["chat", "username"]
        missing = [k for k in required if k not in r]
        if missing:
            return jsonify({"error": f"Missing required keys: {', '.join(missing)}"}), 400

        api_key = request.headers.get("X-API-KEY")
        if not api_key:
            return jsonify({"error": "Missing X-API-KEY header"}), 400

        # === Validate the provided API key for the given user email ===
        
        assert_valid_api_key(api_key, r["username"])
        
        # === PARAMETERS WITH FALLBACK ===
        
        chat = r["chat"]
        if not isinstance(chat, list) or not chat:
            return jsonify({"error": "Invalid 'chat': expected non-empty list"}), 400

        # namespace è usato solo per prompt/logging; il retriever usa RAG_NAMESPACE_DINO
        namespace = r.get("namespace") or os.getenv("RAG_DEFAULT_NAMESPACE", "Dino")
        language = r.get("language") or "ITA"
        token_cost = int(os.getenv("COMPLETION_TOKEN_COST", "1"))

        app.logger.info(
            f"[agentchat] user={r['username']} ns={namespace} lang={language}"
        )

        # === TOKEN CHECK ===
        user_tokens = database_pg.get_user_tokens(r["username"])
        if user_tokens is None:
            return jsonify({"error": "Could not retrieve user tokens"}), 500
        if token_cost > user_tokens:
            return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 403
        
        # === DYNAMIC PROMPT (COMPASS AI TUTOR) ===

        default_agentchat_prompt = textwrap.dedent("""\
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


        # === MODEL AND PROVIDER NORMALIZATION ===

        provider = os.getenv("COMPLETION_MODEL_PROVIDER", "Deepinfra").strip()
        configured_model = os.getenv("COMPLETION_MODEL_AGENT_CHAT", "").strip()

        if not configured_model:
            raise ValueError("Environment variable COMPLETION_MODEL_AGENT_CHAT is not set.")

        # Model used in database logs and cost accounting
        model_clean = configured_model

        # Model passed to LiteLLM (provider prefix required)
        model_id_for_llm = f"{provider.lower()}/{configured_model}"


        # === MODEL AND TOOL INITIALIZATION ===

        # Determine the API key based on the provider
        provider_env_var_map = {
            "Deepinfra": "DEEPINFRA_API_KEY",
            "Mistral": "MISTRAL_API_KEY",
            "Google": "GOOGLE_API_KEY",
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Groq": "GROQ_API_KEY",
        }
        
        # Default fallback to standard naming convention {PROVIDER}_API_KEY
        env_var_name = provider_env_var_map.get(provider, f"{provider.upper()}_API_KEY")
        provider_api_key = os.getenv(env_var_name)
        
        # If specific key not found, try fallback to DEEPINFRA if provider is Deepinfra (redundant but safe) 
        # or logging a warning could be useful, but for now we follow the pattern.

        llm = LiteLLMModel(
            model_id=model_id_for_llm, 
            api_key=provider_api_key,
            temperature=0,
        )        
        
        retriever_tool = RetrieverTool(namespace)

        agent = CodeAgent(
            tools=[retriever_tool],
            model=llm,
            max_steps=5,
            additional_authorized_imports=["json"],
        )
                
        # EXTRACTING LAST USER MESSAGE
        user_message = chat[-1]

        # Dynamic prompt retrieval:
        base_prompt_template = load_prompt(
            "compass_agentchat_system",
            default_text=default_agentchat_prompt
        )

        # Rendering the prompt with dynamic variables (namespace, user_question, language)
        system_prompt = render_prompt(
            base_prompt_template,
            namespace=namespace,
            user_question=user_message,
            language=language
        )
        
        # === AGENT EXECUTION AND TIME MEASUREMENT ===
        start_time = time.time()
        app.logger.info(f"[agentchat] Executing agent message='{user_message[:60]}...'")

        result = agent.run(
            user_message,
            additional_args={
                "system_prompt": system_prompt
            },
            return_full_result=True,
        )

        duration_ms = round((time.time() - start_time) * 1000, 2)
        app.logger.debug(f"[agentchat] Run completed in {duration_ms} ms")

        # === RESULT SERIALIZATION ===

        payload = serialize_runresult(result)
        payload["metrics"]["duration_ms"] = duration_ms  # aggiunta durata misurata lato Flask

        # === STRUCTURED LOGGING ===

        log_runresult(
            result,
            user=r["username"],
            namespace=namespace,
            language=language,
            question=user_message,
        )

        log_id: Optional[int] = None

        # === DATABASE TOKEN USAGE LOGGING ===

        try:
            # Retrieve user_id from username
            user = database_pg.get_user_by_username(r["username"])
            if not user:
                raise ValueError(f"User '{r['username']}' not found in DB")

            user_id = user.get("id")
            if not isinstance(user_id, int):
                raise TypeError(f"Invalid user_id: {user_id}")

            # Extract token usage from payload
            token_metrics = payload.get("metrics", {}).get("token_usage", {})
            token_input = token_metrics.get("input", 0)
            token_output = token_metrics.get("output", 0)

            # Use clean model name from earlier normalization
            model = model_clean     

            # Log into PostgreSQL
            log_id = log_token_usage(
                user_id=user_id,
                token_input=token_input,
                token_output=token_output,
                model=model,
                provider=provider,
            )

        except Exception as e:
            app.logger.error(f"[agentchat] Failed to log token usage: {e}")


        # === TOKEN MANAGEMENT ===

        answer_text = payload.get("answer", "")
        if answer_text:
            edit_tokens(r["username"], -token_cost)


        app.logger.info(
            f"[agentchat] done user={r['username']} duration={duration_ms}ms "
            f"tools={len(payload.get('tool_calls', []))} "
            f"vectors={len(payload.get('vectors', []))} "
            f"fu={len(payload.get('follow_ups', []))}"
        )

        if log_id is not None:
            payload["log_id"] = log_id

        return jsonify(payload), 200

    except Exception as e:
        import traceback
        app.logger.error(f"[agentchat] Unexpected error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred"}), 500


@app.route("/prompt.txt", methods=["POST"])
def prompt_handler() -> Union[str, tuple[str, int, dict[str, str]]]:
    prompt = request.form.get("prompt")
    username = request.form.get("username")
    api_key = request.headers.get("X-API-KEY")


# Define a route for the '/admin/costs' endpoint
@app.route("/admin/costs", methods=["GET"])
@admin_required
def admin_costs() -> str:
    costs = get_all_costs()
    return render_template("admin/costs.html", costs=costs)


# Define a route for the '/admin/costs/add' endpoint
@app.route("/admin/costs/add", methods=["POST"])
@admin_required
def admin_add_cost() -> Response:
    model = request.form.get("model")
    provider = request.form.get("provider")
    token_input_cost = float(request.form.get("token_input_cost"))
    token_output_cost = float(request.form.get("token_output_cost"))
    start_date_valid = request.form.get("start_date_valid")
    end_date_valid = request.form.get("end_date_valid")

    error = add_cost(
        model,
        provider,
        token_input_cost,
        token_output_cost,
        start_date_valid,
        end_date_valid,
    )
    if error:
        flash(error, "danger")
    else:
        flash("Cost added successfully", "success")
    return redirect(url_for("admin_costs"))


# Define a route for the '/admin/costs/edit/<int:cost_id>' endpoint
@app.route("/admin/costs/edit/<int:cost_id>", methods=["GET", "POST"])
@admin_required
def admin_edit_cost(cost_id: int) -> str | Response:
    if request.method == "POST":
        model = request.form.get("model")
        provider = request.form.get("provider")
        token_input_cost = float(request.form.get("token_input_cost"))
        token_output_cost = float(request.form.get("token_output_cost"))
        start_date_valid = request.form.get("start_date_valid")
        end_date_valid = request.form.get("end_date_valid")

        error = update_cost(
            cost_id,
            model,
            provider,
            token_input_cost,
            token_output_cost,
            start_date_valid,
            end_date_valid,
        )
        if error:
            flash(error, "danger")
        else:
            flash("Cost updated successfully", "success")
        return redirect(url_for("admin_costs"))

    cost = get_cost_by_id(cost_id)
    if not cost:
        flash("Cost not found", "danger")
        return redirect(url_for("admin_costs"))
    return render_template("admin/edit_cost.html", cost=cost)


# Define a route for the '/admin/costs/delete/<int:cost_id>' endpoint
@app.route("/admin/costs/delete/<int:cost_id>", methods=["POST"])
@admin_required
def admin_delete_cost(cost_id: int) -> Response:
    error = delete_cost(cost_id)
    if error:
        flash(error, "danger")
    else:
        flash("Cost deleted successfully", "success")
    return redirect(url_for("admin_costs"))

    if not api_key:
        return "Missing API key", 400, textContentType
    if not prompt:
        return "No prompt provided", 400, textContentType
    if not username:
        return "Username not provided", 400, textContentType

    assert_valid_api_key(api_key, username)

    user_tokens = database_pg.get_user_tokens(username)
    if user_tokens is None:
        return "Could not retrieve user tokens", 500, textContentType

    token_cost = int(PROMPT_TOKEN_COST or "1")
    if token_cost > user_tokens:
        return f"Not enough tokens, user_tokens: {user_tokens}", 400, textContentType

    model_name = PROMPT_MODEL or "gpt-3.5-turbo"
    llm_type = PROMPT_PROVIDER or "openai"

    try:
        resp = reply_to_prompt(prompt, username, llm_type, model_name)
        return resp, 200, textContentType
    except Exception as e:
        return str(e), 500, textContentType


@app.route("/storeragfile", methods=["POST"])
def store_rag_file() -> tuple[str, int, dict[str, str]]:
    graphql_url = request.form.get("graphqlUrl")
    auth_token = request.form.get("authToken")
    err = dino_authenticate(graphql_url, auth_token)
    if err:
        return str(err), 403, textContentType

    file = request.files.get("file")
    url = request.form.get("url")
    namespace = request.form.get("namespace") or ""

    if not file:
        return "File not provided", 400, textContentType
    if not url:
        return "Url not provided", 400, textContentType

    chunk_size = 900
    chunk_overlap = 100
    tx_split = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    md_split = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    metadata = {"url": url, "mimetype": file.mimetype, "source": file.filename}
    text = ""
    paragraphs: List[Document] = []
    try:
        text = ""
        is_markdown = False

        if file.mimetype == "text/plain":
            text = file.stream.read().decode()
            paragraphs = tx_split.split_documents(
                [Document(page_content=text, metadata=metadata)]
            )
        elif file.mimetype == "text/markdown":
            text = file.stream.read().decode()
            paragraphs = md_split.split_documents(
                [Document(page_content=text, metadata=metadata)]
            )
        elif file.mimetype == "application/pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp:
                file.save(temp.name)
                # Method 'to_markdown' of library 'pymupdf4llm' incorrectly hints always returning a string (return a List[dict] in this case)
                pages: List[dict] = pymupdf4llm.to_markdown(temp.name, page_chunks=True)  # type: ignore
                page_texts = [p["text"] for p in pages]
                text = "".join(page_texts)
                page_docs = [
                    Document(
                        page_content=p["text"],
                        metadata=metadata | {"page": p["metadata"]["page"]},
                    )
                    for p in pages
                ]
                paragraphs = md_split.split_documents(page_docs)
        elif file.mimetype.startswith("audio"):
            resp = whisper_response(file)
            if resp.status_code != 200:
                return "Error whispering audio", 500, textContentType
            json = resp.json()
            text = json["text"]
            segments = [
                Document(
                    page_content=s["text"],
                    metadata=metadata | {"start_time": s["start"]},
                )
                for s in json["segments"]
            ]
            paragraphs = merge_segments(segments, chunk_size)
        elif file.mimetype.startswith("image"):
            text = describe_image(url, VISION_PROVIDER or "", VISION_MODEL or "")
        else:
            return "Unsupported file type", 400, textContentType

        if text == "":
            return "", 200, textContentType

        embeddings = choose_emb_model(
            COMPLETION_EMBEDDING_MODEL_PROVIDER or "", COMPLETION_EMBEDDING_MODEL or ""
        )
        store = PGVectorStore(embeddings, namespace)
        store.store_paragraphs(paragraphs)
        return text, 200, textContentType

    except Exception as e:
        return str(e), 500, textContentType


def whisper_response(file):
    url = f"https://api.deepinfra.com/v1/inference/{WHISPER_MODEL}"
    headers = {"Authorization": f"bearer {DEEPINFRA_API_KEY}"}
    files = {"audio": file, "response_format": (None, "text")}
    return requests.post(url, headers=headers, files=files)


# Define a route for the '/transcribe' endpoint
@app.route("/transcribe", methods=["POST"])
def whisper_parse() -> Union[Response, tuple[Response, int]]:
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    user_name_header = request.headers.get("X-USER-NAME")

    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400
    if not user_name_header:
        return jsonify({"error": "Missing X-USER-NAME header"}), 400

    user_name = user_name_header.replace(" ", "_").strip()
    assert_valid_api_key(api_key, user_email)

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Missing file"}), 400

    lang = request.form.get("lang") or "ENG"

    if file.mimetype.startswith("audio"):
        response = whisper_response(file)
        if response.status_code == 200:
            try:
                return jsonify(response.json()), 200
            except Exception as e:
                return jsonify({"error": f"Invalid JSON from whisper: {str(e)}"}), 500
        else:
            app.logger.error(f"Whisper failed: {response.status_code} - {response.text}")
            return jsonify({"error": "Whisper transcription failed"}), 500

    if file.mimetype == "application/pdf":
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp:
                file.save(temp.name)
                text = pymupdf4llm.to_markdown(temp.name)
                return jsonify({"text": text}), 200
        except Exception as e:
            return jsonify({"error": f"Error extracting text from pdf: {str(e)}"}), 422
        
    if file.mimetype.startswith("image"):
        try:
            b64 = base64.b64encode(file.read()).decode()
            dataurl = f"data:{file.mimetype};base64,{b64}"
            text = describe_image(dataurl, VISION_PROVIDER or "", VISION_MODEL or "")
            return jsonify({"text": text}), 200
        except Exception as e:
            return jsonify({"error": f"Error extracting text from image: {str(e)}"}), 500

    return jsonify({"error": f"Unexpected file mimetype: {file.mimetype}"}), 400


# Define a route for the '/audioformcompilation' endpoint
@app.route("/audioformcompilation", methods=["POST"])
def audio_form_compile() -> Union[Response, tuple[Response, int]]:
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")

    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    assert_valid_api_key(api_key, user_email)

    if not request.json:
        return jsonify({"error": "Missing JSON body"}), 400

    formSchemaName = request.json.get("name")
    formSchemaExampleData = request.json.get("exampledata")
    formSchemaChoices = request.json.get("choices")
    transcribedAudio = request.json.get("transcribedAudio")

    if not formSchemaExampleData:
        return jsonify({"error": "Missing Schema example empty data"}), 400
    if not formSchemaName:
        return jsonify({"error": "Missing Schema Name"}), 400
    if not transcribedAudio:
        return jsonify({"error": "Missing Transcribed Audio"}), 400

    user_tokens = database_pg.get_user_tokens(user_email)
    if user_tokens is None:
        return jsonify({"error": "Could not retrieve user tokens"}), 500

    token_cost = int(AUDIO_FORM_TOKEN_COST or "1")
    if token_cost > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    model_name = AUDIO_MODEL or "gpt-3.5-turbo"
    llm_type = AUDIO_PROVIDER or "openai"

    prompts = audioFormPromptBuild(
        formSchemaExampleData,
        formSchemaName,
        formSchemaChoices,
        transcribedAudio,
    )

    if not prompts:
        return jsonify({"error": "Failed to build prompts"}), 500

    invocation = audioFormCompilation(
        prompts["userprompt"],
        prompts["systemprompt"],
        user_email,
        llm_type,
        model_name,
    )

    if invocation:
        edit_tokens(user_email, -token_cost)

    app.logger.debug(f"Audio form compilation result: {invocation}")
    return jsonify(invocation), 200

# Define a route for the '/summarize' endpoint that returns a "not yet implemented" message
@app.route("/summarize", methods=["GET"])
def summarize():
    return "The /summarize endpoint is not yet implemented.", 501


# Define a route for the '/summarize' endpoint that returns a "not yet implemented" message
@app.route("/categorize", methods=["GET"])
def categorize():
    return "The /categorize endpoint is not yet implemented.", 501


# Define a route for the '/img-comparison' endpoint that returns a "not yet implemented" message
@app.route("/img-comparison", methods=["GET"])
def img_comparison():
    return "The /img-comparison endpoint is not yet implemented.", 501




@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password and bcrypt.checkpw(password.encode('utf-8'), ADMIN_PASSWORD_HASH):
            session['admin_logged_in'] = True
            session['admin_username'] = username
            flash('Successfully logged in!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials', 'danger')
    
    return render_template('admin/login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    flash('Successfully logged out', 'info')
    return redirect(url_for('admin_login'))

@app.route('/admin')
@admin_required
def admin_dashboard():
    env_vars = {
        "DATACHAT_MODEL": DATACHAT_MODEL,
        "DATACHAT_PROVIDER": DATACHAT_PROVIDER,
        "PROMPT_MODEL": PROMPT_MODEL,
        "PROMPT_PROVIDER": PROMPT_PROVIDER,
        "AUDIO_MODEL": AUDIO_MODEL,
        "AUDIO_PROVIDER": AUDIO_PROVIDER,
        "COMPLETION_MODEL": COMPLETION_MODEL,
        "COMPLETION_MODEL_PROVIDER": COMPLETION_MODEL_PROVIDER,
        "COMPLETION_MODEL_AGENT_CHAT": COMPLETION_MODEL_AGENT_CHAT, 
        "COMPLETION_EMBEDDING_MODEL": COMPLETION_EMBEDDING_MODEL,
        "COMPLETION_EMBEDDING_MODEL_PROVIDER": COMPLETION_EMBEDDING_MODEL_PROVIDER,
        "WHISPER_MODEL": WHISPER_MODEL,
        "VISION_PROVIDER": VISION_PROVIDER,
        "VISION_MODEL": VISION_MODEL,
        "DATACHAT_TOKEN_COST": DATACHAT_TOKEN_COST,
        "COMPLETION_TOKEN_COST": COMPLETION_TOKEN_COST,
        "PROMPT_TOKEN_COST": PROMPT_TOKEN_COST,
        "AUDIO_FORM_TOKEN_COST": AUDIO_FORM_TOKEN_COST
    }
    
    try:
        stats_data = get_users_stats()
        
        today = datetime.now().strftime("%Y-%m-%d")
        daily_stats = get_daily_stats(today)
        recent_activity = get_recent_activity()
        
        stats = {
            'total_users': stats_data['total_users'],
            'active_sessions': stats_data['total_tokens'], # Keeping this for now, but will replace in template
            'daily_tokens': daily_stats['total_tokens'],
            'daily_cost': daily_stats['total_cost'],
            'total_orders': 0,
            'recent_activity': recent_activity,
            'db_connected': True,  # If we got here, DB is connected
  
            "cpu_percent": psutil.cpu_percent(interval=0.5),  # utilizzo CPU
            "memory": {
                "total": psutil.virtual_memory().total,
                "used": psutil.virtual_memory().used,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            }
        }
        return render_template('admin/dashboard.html', stats=stats, env_vars=env_vars)
        
    except Exception as e:
        flash(f'Errore nel caricamento dashboard: {str(e)}', 'danger')
        stats = {
            'total_users': 0, 
            'active_sessions': 0, 
            'total_orders': 0,
            'db_connected': False,  # DB connection failed
            'recent_activity': [],
            'daily_tokens': 0,
            'daily_cost': 0.0,
            'cpu_percent': 0,
            'memory': {
                'total': 0,
                'used': 0,
                'available': 0,
                'percent': 0
            }
        }
        
        return render_template('admin/dashboard.html', stats=stats, env_vars=env_vars)

@app.route('/admin/users')
@admin_required
def admin_users():
    try:
        users = get_users_for_admin()
        return render_template('admin/users.html', users=users)
        
    except Exception as e:
        flash(f'Errore nel recupero utenti: {str(e)}', 'danger')
        return render_template('admin/users.html', users=[])

@app.route('/admin/logs')
@admin_required
def admin_logs():
    try:
        logs = get_logs_for_admin(limit=100)
        stats = get_logs_stats()
        return render_template('admin/logs.html', logs=logs, stats=stats)
        
    except Exception as e:
        flash(f'Errore nel recupero logs: {str(e)}', 'danger')
        return render_template('admin/logs.html', logs=[], stats={})


@app.route('/admin/feedback')
@admin_required
def admin_feedback():
    try:
        source_filter = request.args.get('source')
        if source_filter == 'all':
            source_filter = None
            
        feedbacks = get_feedback_for_admin(source_filter)
        stats = get_feedback_stats(source_filter)
        
        return render_template(
            'admin/feedback.html', 
            feedbacks=feedbacks, 
            stats=stats,
            current_filter=source_filter
        )
    except Exception as e:
        flash(f'Errore nel recupero feedback: {str(e)}', 'danger')
        return render_template('admin/feedback.html', feedbacks=[], stats={})


@app.route('/admin/users/<int:user_id>/edit', methods=['GET', 'POST'])
@admin_required
def admin_edit_user(user_id):
    if request.method == 'POST':
        try:
            new_tokens = request.form.get('tokens', type=int)
            
            if new_tokens is None or new_tokens < 0:
                flash('Numero di token non valido', 'danger')
                return redirect(url_for('admin_users'))
            
            success = update_user_tokens(user_id, new_tokens)
            
            if success:
                flash(f'Token aggiornati con successo a {new_tokens}', 'success')
            else:
                flash('Utente non trovato', 'danger')
                
        except Exception as e:
            flash(f"Errore nell'aggiornamento: {str(e)}", 'danger')
        
        return redirect(url_for('admin_users'))
    
    # GET request - show edit form
    try:
        user = get_user_by_id(user_id)
        if user:
            return render_template('admin/edit_user.html', user=user)
        else:
            flash('Utente non trovato', 'danger')
            return redirect(url_for('admin_users'))
    except Exception as e:
        flash(f'Errore: {str(e)}', 'danger')
        return redirect(url_for('admin_users'))

@app.route('/admin/prompts')
@admin_required
def admin_prompts():
    try:
        prompts = get_all_prompts()
        return render_template('admin/prompts.html', prompts=prompts)
    except Exception as e:
        flash(f'Errore nel recupero prompt: {str(e)}', 'danger')
        return render_template('admin/prompts.html', prompts=[])

@app.route('/admin/prompts/add', methods=['POST'])
@admin_required
def admin_add_prompt():
    try:
        title = request.form.get('title')
        version = request.form.get('version', type=int)
        message = request.form.get('message')
        
        if not title or not version or not message:
            flash('Tutti i campi sono obbligatori', 'danger')
            return redirect(url_for('admin_prompts'))
        
        add_prompt(title, version, message)
        flash('Prompt aggiunto con successo', 'success')
    except Exception as e:
        flash(f'Errore nell\'aggiunta del prompt: {str(e)}', 'danger')
    
    return redirect(url_for('admin_prompts'))

@app.route('/admin/prompts/<int:prompt_id>/edit', methods=['GET', 'POST'])
@admin_required
def admin_edit_prompt(prompt_id):
    if request.method == 'POST':
        try:
            title = request.form.get('title')
            version = request.form.get('version', type=int)
            message = request.form.get('message')
            
            if not title or not version or not message:
                flash('Tutti i campi sono obbligatori', 'danger')
                return redirect(url_for('admin_edit_prompt', prompt_id=prompt_id))
            
            success = update_prompt(prompt_id, title, version, message)
            
            if success:
                flash('Prompt aggiornato con successo', 'success')
            else:
                flash('Prompt non trovato', 'danger')
                
        except Exception as e:
            flash(f"Errore nell'aggiornamento: {str(e)}", 'danger')
        
        return redirect(url_for('admin_prompts'))
    
    # GET request - show edit form
    try:
        prompt = get_prompt_by_id(prompt_id)
        if prompt:
            return render_template('admin/edit_prompt.html', prompt=prompt)
        else:
            flash('Prompt non trovato', 'danger')
            return redirect(url_for('admin_prompts'))
    except Exception as e:
        flash(f'Errore: {str(e)}', 'danger')
        return redirect(url_for('admin_prompts'))

@app.route('/admin/prompts/<int:prompt_id>/delete', methods=['POST'])
@admin_required
def admin_delete_prompt(prompt_id):
    try:
        success = delete_prompt(prompt_id)
        if success:
            flash('Prompt eliminato con successo', 'success')
        else:
            flash('Prompt non trovato', 'danger')
    except Exception as e:
        flash(f"Errore nell'eliminazione: {str(e)}", 'danger')
    
    return redirect(url_for('admin_prompts'))

@app.route("/health")
def health():
    # Stato base
    status = {
        "status": "ok",
    }
    return jsonify(status)


if __name__ == "__main__":
    app.run(debug=True,port=8000)
