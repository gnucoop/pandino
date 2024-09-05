# Import necessary libraries for the Flask application
import os
from flask import Flask, request, jsonify, abort
import pandas as pd
from pandasai import Agent
import database
from database import validate_api_key
from agent_manager import getAgent, createAgent, deleteAgent
from file_manager import base64toFile, isImageFilePath, fileToBase64

# Import specific chat models from their respective libraries
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

# Initialize the Flask application
app = Flask(__name__)


# Define a route for the '/' endpoint that returns a welcome message
@app.route("/")
def welcome():
    return "Welcome to Pandino! This is the root endpoint."


def validate_api_key(api_key):
    if not api_key:
        abort(403)
    result, message = database.validate_api_key(api_key)
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
    
    result, message = database.validate_api_key(api_key)

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

    # Extract necessary parameters from the request JSON
    model_name = request.json.get("model_name")
    llm_type = request.json.get("llm_type")

    # Check if all required parameters are present
    if not model_name or not llm_type or not user_name:
        return jsonify({"error": "Missing parameters"}), 400

    # Extract the data parameter from the request JSON
    data_param = request.json.get("data")
    if not data_param:
        return jsonify({"error": "Missing data parameter"}), 400

    # Read the data from the provided CSV file
    fileData = base64toFile(data_param)
    data = pd.read_csv(fileData, sep=",")

    # Initialize the language model based on the provided type
    if llm_type == "Groq":
        model_kwargs = {"seed": 26}
        llm = ChatGroq(
            model_name=model_name,
            temperature=0,
            api_key=os.environ["GROQ_API_KEY"],
            model_kwargs=model_kwargs,
        )
    elif llm_type == "Deepseek":
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            seed=26,
            base_url="https://api.deepseek.com",
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )
    elif llm_type == "Mistral":
        llm = ChatMistralAI(
            model_name=model_name,
            temperature=0,
            seed=26,
            api_key=os.environ["MISTRAL_API_KEY"],
        )
    elif llm_type == "OpenAI":
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            seed=26,
            api_key=os.environ["OPENAI_API_KEY"],
        )

    # Initialize the agent with the data and configuration
    # agent = Agent(data, config={"llm": llm, "open_charts": False})
    try:
        agent = createAgent(api_key, data, llm, user_name)
        return jsonify({"Agent active": agent.conversation_id})
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
        response_dict = response.to_dict(orient="records")
    elif isinstance(response, dict):
        response_dict = response
    else:
        response_dict = {"type": type(response).__name__, "value": str(response)}
        if response_dict and response_dict["value"]:
            # Convert image file path in value to a base64 serialized file
            if isImageFilePath(response_dict["value"]):
                response_dict["type"] = "image"
                response_dict["value"] = fileToBase64(response_dict["value"])

    return jsonify({"response": response_dict})


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
