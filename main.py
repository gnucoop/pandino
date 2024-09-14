# Import necessary libraries for the Flask application
import os
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import pandas as pd
from pandasai import Agent
from agent_manager import getAgent, createAgent, deleteAgent
from file_manager import isImageFilePath, fileToBase64

# Import specific chat models from their respective libraries
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

# Import function from ai and database
import databasepg
from databasepg import validate_api_key
import dino
from dino import dino_authenticate
import ai
from ai import complete_chat, CompletionResponse
from ai import reply_to_prompt, choose_llm

# Initialize the Flask application
app = Flask(__name__)
# origins=["http://localhost:4200"]
CORS(app)

# Removing Pandas read csv columns limitations to avoid truncated dataFrames
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


# Define a route for the '/' endpoint that returns a welcome message
@app.route("/")
def welcome():
    return "Welcome to Pandino! This is the root endpoint."


def validate_api_key(api_key):
    if not api_key:
        abort(403)
    result, message = databasepg.validate_api_key(api_key)
    if not result:
        if "expired" in message:
            abort(403, description="API key expired")
        else:
            abort(403, description="Invalid API key")


# Define a route for the '/validateapikey' endpoint that accepts POST requests
@app.route("/validateapikey", methods=["POST"])
def validate():
    api_key = request.headers.get("X-API-KEY")

    # Check if all required parameters are present
    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400

    result, message = databasepg.validate_api_key(api_key)

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
    user_name_header = request.headers.get("X-USER-NAME")
    user_name = (
        user_name_header.replace(" ", "_").strip() if user_name_header != None else None
    )
    validate_api_key(api_key)

    # Check if all required parameters are present
    if not user_name:
        return jsonify({"error": "Missing X-USER-NAME header"}), 400

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
    user_name = user_name_header.replace(" ", "_").strip()
    validate_api_key(api_key)

    # Extract necessary parameters from the request FORMDATA
    request_model_name = request.form.get("model_name")
    request_llm_type = request.form.get("llm_type")
    request_file = request.files.get("file")
    model_name = request_model_name if request_model_name else "llama-3.1-70b-versatile"
    llm_type = request_llm_type if request_llm_type else "Groq"

    # Check if all required parameters are present
    if not model_name or not llm_type or not user_name or not request_file:
        return jsonify({"error": "Missing parameters"}), 400

    # Read the data from the provided CSV file
    data = pd.read_csv(request_file, sep=",")
    # Initialize the language model based on the provided type
    llm = choose_llm(llm_type, model_name)

    # Initialize the agent with the data and configuration
    try:
        agent = createAgent(api_key, data, llm, user_name)
        agentResponse = {"Agent active": agent.conversation_id}

        suggestionsQuestion = f"""Given this pandas dataframe {data}: Try to understand the nature of the data and suggest me what kind of analysis should I ask for. 
        Explain in details your answers and do any suggestions of possible question that I could ask. 
        DO not suggest any python code. Please reply in a readable html format, with no asterisks and adding a line break after each paragraph."""
        suggestionsResponse = llm.invoke(suggestionsQuestion)
        if suggestionsResponse and suggestionsResponse.content is not None:
            agentResponse.update({"suggested_questions": suggestionsResponse.content})

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
    validate_api_key(api_key)
    chat = request.json.get("chat")
    agent: Agent | None = getAgent(api_key)

    # Check if the Chat parameter is present
    if not chat:
        return jsonify({"error": "Missing Chat string"}), 400

    # Check if the Agent is active
    if not agent:
        return jsonify({"error": "Agent not active for this Api Key"}), 400

    # Perform the chat operation and get the response and explanation
    response = agent.chat(chat)
    explanation = agent.explain()

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
            "value": response.to_dict(orient="records"),
        }
    elif isinstance(response, dict):
        response_dict = response
        response_dict.update({"type": "dict"})
    else:
        response_dict = {"type": type(response).__name__, "value": str(response)}
        if response_dict and response_dict["value"]:
            # Convert image file path in value to a base64 serialized file
            if isImageFilePath(response_dict["value"]):
                response_dict["type"] = "image"
                response_dict["value"] = fileToBase64(response_dict["value"])

    return jsonify({"response": response_dict, "explanation": explanation})

@app.route("/completion.json", methods=["POST"])
def completion_handler():
    try:
        r = request.get_json()
        if not r:
            return jsonify({"error": "No JSON data provided"}), 400

        required_keys = ["dinoGraphql", "authToken", "chat"]
        missing_keys = [key for key in required_keys if key not in r]

        if missing_keys:
            return jsonify({"error": f"Missing required keys: {', '.join(missing_keys)}"}), 400

        err = dino_authenticate(r["dinoGraphql"], r["authToken"])
        if err:
            return jsonify({"error": f"Authentication error: {str(err)}"}), 401

        # Prepare the request for complete_chat
        chat_request = ai.CompletionRequest(
            dino_graphql=r["dinoGraphql"],
            auth_token=r["authToken"],
            namespace=r.get("namespace", ""),
            info=r.get("info", []),
            chat=r["chat"]
        )

        resp = complete_chat(chat_request)
        
        if isinstance(resp, CompletionResponse):
            if resp.error:
                return jsonify({"error": f"Chat completion error: {resp.error}"}), 400
            return jsonify({"answer": resp.answer,"paragraphs": resp.paragraphs, "similarities":resp.similarities})
        elif resp is None:
            return jsonify({"error": "No response from chat completion"}), 500
        else:
            return jsonify({"error": "Unexpected response format"}), 500

    except Exception as e:
        app.logger.error(f"Unexpected error in completion_handler: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/prompt.txt', methods=['POST'])
def prompt_handler():
    graphql_url = request.form.get('graphqlUrl')
    auth_token = request.form.get('authToken')
    prompt = request.form.get('prompt')

    if not graphql_url or not auth_token:
        return jsonify({'error': 'Auth parameters not provided'}), 400
    
    try:
        # Assuming DinoAuthenticate is replaced with a similar function in Python
        dino_authenticate(graphql_url, auth_token)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
   
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        resp = reply_to_prompt(prompt)
        if isinstance(resp, CompletionResponse):
            return jsonify({
                'answer': resp.answer
            })
        else:
            return jsonify({'error': 'Unexpected response format'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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


# Run the Flask application in debug mode if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True)
