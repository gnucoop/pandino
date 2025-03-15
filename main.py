# Import necessary libraries for the Flask application
import math
import os
import warnings
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import pandas as pd
from pandasai import Agent
import requests
from agent_manager import getAgent, createAgent, deleteAgent
from file_manager import isImageFilePath, fileToBase64
import matplotlib
import secrets
from datetime import datetime

matplotlib.use("Agg")  # Use non-interactive backend
import os

# Import specific chat models from their respective libraries
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Import function from ai and database
import database_pg
from database_pg import edit_tokens, validate_api_key
# import database_pg as database_pg
# from database_pg import edit_tokens, validate_api_key
from dino import dino_authenticate
import ai
from ai import audioFormCompilation, audioFormPromptBuild, complete_chat, CompletionResponse, sentiment_analysis, summarize_text
from ai import reply_to_prompt, choose_llm

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Initialize the Flask application
app = Flask(__name__)
# origins=["http://localhost:4200"]
CORS(app)

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
COMPLETION_MODEL = os.environ.get("COMPLETION_MODEL")
COMPLETION_MODEL_PROVIDER = os.environ.get("COMPLETION_MODEL_PROVIDER")
COMPLETION_EMBEDDING_MODEL = os.environ.get("COMPLETION_EMBEDDING_MODEL")
COMPLETION_EMBEDDING_MODEL_PROVIDER = os.environ.get(
    "COMPLETION_EMBEDDING_MODEL_PROVIDER"
)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL")
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
SA_PROVIDER = os.environ.get("SA_PROVIDER")
SA_MODEL = os.environ.get("SA_MODEL")
STRIPE_KEY = os.environ.get("STRIPE_SK_KEY")
DATACHAT_TOKEN_COST = os.environ.get("DATACHAT_TOKEN_COST")
COMPLETION_TOKEN_COST = os.environ.get("COMPLETION_TOKEN_COST")
PROMPT_TOKEN_COST = os.environ.get("PROMPT_TOKEN_COST")

# Define a route for the '/' endpoint that returns a welcome message
@app.route("/")
def welcome():
    return "Welcome to Pandino! This is the root endpoint."


# Validates an API Key associated to an user email
def validate_api_key(api_key, user_email):
    if not api_key:
        abort(403)
    result, message = database_pg.validate_api_key(api_key, user_email)
    if not result:
        if "expired" in message:
            abort(403, description="API key expired")
        else:
            abort(403, description="Invalid API key")


# Tries to authenticate a Dino user against their Dino instance's backend
def authenticate_dino(graphql_url, auth_token):
    if not graphql_url or not auth_token:
        abort(403)
    result = dino_authenticate(graphql_url, auth_token)
    if result:
        abort(403, description=result)

# Recursively replace NaN with None in dictionaries or lists.
def replace_nan(data):
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
def editTokens():
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


# Define a route for the '/edittokens' endpoint that accepts POST requests
@app.route("/getusertokens", methods=["POST"])
def getUserTokens():
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400
    validate_api_key(api_key, user_email)
    tokens = database_pg.get_user_tokens(user_email)
    return jsonify({"response": {"tokens": tokens}}), 200


# Define a route for the '/adduser' endpoint that accepts POST requests
@app.route("/checkpandinouser", methods=["POST"])
def addNewUser():
    graphql_url = request.headers.get("X-GRAPHQL-URL")
    auth_token = request.headers.get("X-AUTH-TOKEN")
    user_email = request.headers.get("X-USER-EMAIL")
    if not graphql_url:
        return jsonify({"error": "Missing X-GRAPHQL-URL header"}), 400
    if not auth_token:
        return jsonify({"error": "Missing X-AUTH-TOKEN header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    try:
        authenticate_dino(graphql_url, auth_token)
    except Exception as e:
        return str(e), 500, {"Content-Type": "text/plain"}

    existingUser = database_pg.get_user_by_username(user_email)
    if not existingUser:
        generatedKey = secrets.token_urlsafe(8)
        currentDate = datetime.now()
        expirationDate = currentDate.replace(year=currentDate.year + 2)
        addUserResult = database_pg.add_user(
            user_email, generatedKey, expirationDate
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
def validate():
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    # Check if all required parameters are present
    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400
    result, message = database_pg.validate_api_key(api_key, user_email)

    if not result:
        if "expired" in message:
            return jsonify({"error": "API key expired"}), 403
        else:
            return jsonify({"error": "Invalid API key"}), 403
    else:
        return jsonify({"response": "API key match found"}), 200


# Define a route for the '/endchat' endpoint that accepts POST requests
@app.route("/enddatachat", methods=["POST"])
def endChat():
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    user_name_header = request.headers.get("X-USER-NAME")
    user_name = (
        user_name_header.replace(" ", "_").strip() if user_name_header != None else None
    )
    validate_api_key(api_key, user_email)

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
def startChat():
    api_key = request.headers.get("X-API-KEY")
    user_name_header = request.headers.get("X-USER-NAME")
    user_email = request.headers.get("X-USER-EMAIL")
    user_name = user_name_header.replace(" ", "_").strip()
    validate_api_key(api_key, user_email)

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
    if int(DATACHAT_TOKEN_COST) > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    # Read the data from the provided CSV file
    data = pd.read_csv(request_file, sep=",")
    # Initialize the language model based on the provided type
    llm = choose_llm(llm_type, model_name)
    # Initialize the agent with the data and configuration
    try:
        agent = createAgent(api_key, data, llm, user_name)
        agentResponse = {"Agent active": agent.conversation_id}

        question_templates = {
            "ITA": f"""Dato questo dataframe pandas {data}. Prova a capire la natura dei dati e suggeriscimi che tipo di analisi dovrei chiedere. Spiega in dettaglio le tue risposte e fai qualsiasi suggerimento di possibile domanda che potrei fare. Non suggerire alcun codice Python. Per favore, rispondi in un formato html leggibile, senza asterischi e aggiungendo un'interruzione di riga dopo ogni paragrafo.""",
            "ENG": f"""Given this pandas dataframe {data}. Try to understand the nature of the data and suggest me what kind of analysis should I ask for. Explain in details your answers and make any suggestions about possible questions that I could ask. DO not suggest any python code. Please reply in a readable html format, with no asterisks and adding a line break after each paragraph.""",
            "FRA": f"""Étant donné ce dataframe pandas {data}. Essayez de comprendre la nature des données et suggérez-moi quel type d'analyse je devrais demander. Expliquez en détail vos réponses et faites toutes les suggestions de questions possibles que je pourrais poser. Ne suggérez aucun code Python. Veuillez répondre dans un format html lisible, sans astérisques et en ajoutant un saut de ligne après chaque paragraphe.""",
            "ESP": f"""Dado este dataframe pandas {data}. Intenta entender la naturaleza de los datos y sugiereme qué tipo de análisis debería preguntar. Explica en detalle tus respuestas y haz cualquier sugerencia de pregunta posible que podría hacer. No sugieras ningún código Python. Por favor, responde en un formato html legible, sin asteriscos y agregando un salto de línea después de cada párrafo.""",
            # Add more languages as needed
        }
        suggestionsQuestion = (
            question_templates.get(lang, question_templates[lang])
            if lang in question_templates
            else question_templates.get("ENG", question_templates["ENG"])
        )
        suggestionsResponse = llm.invoke(suggestionsQuestion)
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
def dataChat():
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    validate_api_key(api_key, user_email)
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
def buyReport():
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    validate_api_key(api_key, user_email)
    prompts = request.json.get("prompts")

    # Check if the Chat parameter is present
    if not prompts or not isinstance(prompts, int):
        return jsonify({"error": "Missing Prompts numeric parameter"}), 400

    # Check if user email is present
    if not user_email:
        return jsonify({"error": "Missing User email"}), 400

    # Checks if the User's tokens are enough for this operation
    user_tokens = database_pg.get_user_tokens(user_email)
    if int(prompts) > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    # Spends User's tokens
    result, message = edit_tokens(user_email, -int(prompts))

    return jsonify({"response": result, "message": f"{message}"}), 200


@app.route("/completion.json", methods=["POST"])
def completion_handler():
    try:
        r = request.get_json()
        if not r:
            return jsonify({"error": "No JSON data provided"}), 400

        required_keys = ["dinoGraphql", "authToken", "chat", "username"]
        missing_keys = [key for key in required_keys if key not in r]

        if missing_keys:
            return (
                jsonify({"error": f"Missing required keys: {', '.join(missing_keys)}"}),
                400,
            )

        api_key = request.headers.get("X-API-KEY")
        validate_api_key(api_key, r["username"])

        # Checks if the User's tokens are enough for this operation
        user_tokens = database_pg.get_user_tokens(r["username"])
        if int(COMPLETION_TOKEN_COST) > user_tokens:
            return (
                jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}),
                500,
            )

        err = dino_authenticate(r["dinoGraphql"], r["authToken"])
        if err:
            return jsonify({"error": f"Authentication error: {str(err)}"}), 401

        # Prepare the request for complete_chat
        chat_request = ai.CompletionRequest(
            dino_graphql=r["dinoGraphql"],
            auth_token=r["authToken"],
            namespace=r.get("namespace", ""),
            username=r["username"],
            info=r.get("info", []),
            chat=r["chat"],
        )

        llm_type = COMPLETION_MODEL_PROVIDER
        model = COMPLETION_MODEL
        emb_llm_type = COMPLETION_EMBEDDING_MODEL_PROVIDER
        emb_model = COMPLETION_EMBEDDING_MODEL

        resp = complete_chat(chat_request, llm_type, model, emb_llm_type, emb_model)
        for vec in resp.vectors:
            vec['similarity'] += 0.3
        if isinstance(resp, CompletionResponse):
            if resp.error:
                return jsonify({"error": f"Chat completion error: {resp.error}"}), 200

            # Spends User's tokens
            if resp.answer or resp.vectors:
                edit_tokens(r["username"], -int(COMPLETION_TOKEN_COST))

            return jsonify({
                "answer": resp.answer,
                "vectors": resp.vectors,
            })
        elif resp is None:
            return jsonify({"error": "No response from chat completion"}), 500
        else:
            return jsonify({"error": "Unexpected response format"}), 500

    except Exception as e:
        app.logger.error(f"Unexpected error in completion_handler: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


@app.route("/prompt.txt", methods=["POST"])
def prompt_handler():
    graphql_url = request.form.get("graphqlUrl")
    auth_token = request.form.get("authToken")
    prompt = request.form.get("prompt")
    username = request.form.get("username")
    model_name = PROMPT_MODEL
    llm_type = PROMPT_PROVIDER

    api_key = request.headers.get("X-API-KEY")

    if not graphql_url or not auth_token:
        return "Auth parameters not provided", 400, {"Content-Type": "text/plain"}

    if not username:
        return "Username not provided", 400, {"Content-Type": "text/plain"}

    validate_api_key(api_key, username)

    # Checks if the User's tokens are enough for this operation
    user_tokens = database_pg.get_user_tokens(username)
    if int(PROMPT_TOKEN_COST) > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    try:
        # Assuming DinoAuthenticate is replaced with a similar function in Python
        dino_authenticate(graphql_url, auth_token)
    except Exception as e:
        return str(e), 500, {"Content-Type": "text/plain"}

    if not prompt:
        return "No prompt provided", 400, {"Content-Type": "text/plain"}

    try:
        resp = reply_to_prompt(prompt, username, llm_type, model_name)
        if isinstance(resp, CompletionResponse):
            return resp.answer, 200, {"Content-Type": "text/plain"}
        else:
            return "Unexpected response format", 500, {"Content-Type": "text/plain"}
    except Exception as e:
        return str(e), 500, {"Content-Type": "text/plain"}

# Define a route for the '/transcribe' endpoint
@app.route("/transcribe", methods=["POST"])
def whisper_parse():
    api_key = request.headers.get("X-API-KEY")
    user_name_header = request.headers.get("X-USER-NAME")
    user_email = request.headers.get("X-USER-EMAIL")
    user_name = user_name_header.replace(" ", "_").strip()
    validate_api_key(api_key, user_email)

    # Extract necessary parameters from the request FORMDATA
    request_file = request.files.get("file")
    request_lang = request.form.get("lang")
    lang = request_lang if request_lang else "ENG"

    if (
        not user_name
        or not user_email
        or not request_file
    ):
        return jsonify({"error": "Missing parameters"}), 400

    # Prepare the request
    url = f'https://api.deepinfra.com/v1/inference/{WHISPER_MODEL}'
    headers = {
        "Authorization": f"bearer {DEEPINFRA_API_KEY}"
    }
    files = {
        'audio': request_file,
        'response_format': (None, 'text')
    }
    print(files)

    # Send the request
    response = requests.post(url, headers=headers, files=files)
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Define a route for the '/audioformcompilation' endpoint
@app.route("/audioformcompilation", methods=["POST"])
def audio_form_compile():
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    validate_api_key(api_key, user_email)

    model_name = PROMPT_MODEL
    llm_type = PROMPT_PROVIDER

    # Extract necessary parameters from the request FORMDATA
    formSchema = request.json.get("schema")
    formSchemaName = request.json.get("name")
    formSchemaExampleData = request.json.get("exampledata")
    formSchemaChoices = request.json.get("choices")
    transcribedAudio = request.json.get("transcribedAudio")

    # Check if the formSchema parameter is present
    if not formSchema:
        return jsonify({"error": "Missing Schema"}), 400
    
    # Check if the formSchemaExampleData parameter is present
    if not formSchemaExampleData:
        return jsonify({"error": "Missing Schema example empty data"}), 400
    
    # Check if the formSchemaName parameter is present
    if not formSchemaName:
        return jsonify({"error": "Missing Schema Name"}), 400

    # Check if the transcribedAudio parameter is present
    if not transcribedAudio:
        return jsonify({"error": "Missing Transcribed Audio"}), 400

    # Check if user email is present
    if not user_email:
        return jsonify({"error": "Missing User email"}), 400
    
    prompts = audioFormPromptBuild(formSchema, formSchemaExampleData, formSchemaName, formSchemaChoices, transcribedAudio)
    invocation = audioFormCompilation(prompts["userprompt"], prompts["systemprompt"], user_email, llm_type, model_name)
    print(invocation)
    return jsonify(invocation) 

# Define a route for the '/sentiment-analysis' endpoint
@app.route("/sentiment-analysis", methods=["POST"])
def sa():
    request_lang = request.form.get("lang")
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    validate_api_key(api_key, user_email)
    lang = request_lang if request_lang else "ENG"
    model = SA_MODEL
    llm_type = SA_PROVIDER
    comments = request.json.get("comments")

    sa_templates = {
            "ITA": f"""Classifica il testo fornito come positivo, negativo o neutro. Rispondi indicando unicamente la categoria che hai trovato""",
            "ENG": f"""Classify the text provided as positive, negative or neutral. Please reply with just the category you found""",
            "FRA": f"""Classez le texte fourni comme positif, négatif ou neutre. Veuillez répondre en indiquant uniquement la catégorie que vous avez trouvée""",
            "ESP": f"""Clasifique el texto proporcionado como positivo, negativo o neutro. Responda sólo con la categoría que haya encontrado""",
            # Add more languages as needed
        }
    sa_prompt = (
        sa_templates.get(lang, sa_templates[lang])
        if lang in sa_templates
        else sa_templates.get("ENG", sa_templates["ENG"])
    )
    resp = sentiment_analysis(sa_prompt, llm_type, model,comments)
    return jsonify(resp)

    


# Define a route for the '/summarize' endpoint that returns a "not yet implemented" message
@app.route("/summarize", methods=["POST"])
def summ():
    request_lang = request.form.get("lang")
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    validate_api_key(api_key, user_email)
    lang = request_lang if request_lang else "ENG"

    full_text = request.json.get("full_text")

    model_name = PROMPT_MODEL
    llm_type = PROMPT_PROVIDER

    summ_templates = {
            "ITA": f"""Sei un esperto scrittore specializzato nel generare sintesi chiare, concise e fedeli al contenuto originale. Il tuo compito è elaborare testi di qualunque tipologia (articoli, saggi, report, ecc.) e produrre riassunti strutturati che ne preservino il significato intrinseco.""",
            "ENG": f"""You are an experienced writer who specializes in generating summaries that are clear, concise and true to the original content. Your job is to draft texts of any type (articles, essays, reports, etc.) and produce structured summaries that preserve their intrinsic meaning.""",
            "FRA": f"""Vous êtes un rédacteur expérimenté, spécialisé dans la production de résumés clairs et concis, fidèles au contenu original. Votre tâche consiste à rédiger des textes de tous types (articles, essais, rapports, etc.) et à produire des résumés structurés qui préservent leur sens intrinsèque.""",
            "ESP": f"""Usted es un redactor con experiencia especializado en la elaboración de resúmenes claros, concisos y fieles al contenido original. Su tarea consiste en redactar textos de todo tipo (artículos, ensayos, informes, etc.) y elaborar resúmenes estructurados que conserven su significado intrínseco.""",
            # Add more languages as needed
        }
    summ_prompt = (
        summ_templates.get(lang, summ_templates[lang])
        if lang in summ_templates
        else summ_templates.get("ENG", summ_templates["ENG"])
    )
    resp = summarize_text(summ_prompt, llm_type, model_name, full_text)
    print(resp)
    return jsonify(resp)


# Define a route for the '/summarize' endpoint that returns a "not yet implemented" message
@app.route("/categorize", methods=["GET"])
def categorize():
    return "The /categorize endpoint is not yet implemented.", 501


# Define a route for the '/img-comparison' endpoint that returns a "not yet implemented" message
@app.route("/img-comparison", methods=["GET"])
def img_comparison():
    return "The /img-comparison endpoint is not yet implemented.", 501


# Run the Flask application in debug mode if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True)
