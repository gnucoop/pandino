# Import necessary libraries for the Flask application
import os
from flask import Flask, request, jsonify, abort
import pandas as pd
from pandasai import Agent
import database
from database import validate_api_key

# Import specific chat models from their respective libraries
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

# Initialize the Flask application
app = Flask(__name__)

def validate_api_key(api_key):
    if not api_key:
        abort(403)
    result, _ = database.validate_api_key(api_key)
    if not result:
        abort(403)

# Define a route for the '/analyst' endpoint that accepts POST requests
@app.route('/analyst', methods=['POST'])
def analyst():
    api_key = request.headers.get('X-API-KEY')
    validate_api_key(api_key)
    # Extract necessary parameters from the request JSON
    model_name = request.json.get('model_name')
    llm_type = request.json.get('llm_type')
    chat = request.json.get('chat')

    # Check if all required parameters are present
    if not model_name or not llm_type or not chat:
        return jsonify({"error": "Missing parameters"}), 400

    # Extract the data parameter from the request JSON
    data_param = request.json.get('data')
    if not data_param:
        return jsonify({"error": "Missing data parameter"}), 400

    # Read the data from the provided CSV file
    data = pd.read_csv(data_param)

    # Initialize the language model based on the provided type
    if llm_type == 'Groq':
        model_kwargs = {'seed': 26}
        llm = ChatGroq(model_name=model_name, temperature=0, api_key=os.environ['GROQ_API_KEY'], model_kwargs=model_kwargs)
    elif llm_type == 'Deepseek':
        llm = ChatOpenAI(model_name=model_name, temperature=0, seed=26, base_url='https://api.deepseek.com', api_key=os.environ['DEEPSEEK_API_KEY'])
    elif llm_type == 'Mistral':
        llm = ChatMistralAI(model_name=model_name, temperature=0, seed=26, api_key=os.environ['MISTRAL_API_KEY'])
    elif llm_type == 'OpenAI':
        llm = ChatOpenAI(model_name=model_name, temperature=0, seed=26, api_key=os.environ['OPENAI_API_KEY'])

    # Initialize the agent with the data and configuration
    agent = Agent(data, config={"llm": llm, "open_charts": False})

    # Perform the chat operation and get the response and explanation
    response = agent.chat(chat)
    explanation = agent.explain()

    # Convert the response to a DataFrame if it's a list
    if isinstance(response, list):
        try:
            response = pd.DataFrame(response)
        except Exception as e:
            return jsonify({"error": f"Failed to convert list to DataFrame: {str(e)}"}), 500

    # Convert the response to a dictionary
    if isinstance(response, pd.DataFrame):
        response_dict = response.to_dict(orient='records')
    elif isinstance(response, dict):
        response_dict = response
    else:
        response_dict = {'type': type(response).__name__, 'value': str(response)}

    return jsonify({"response": response_dict, "explanation": explanation})

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)
