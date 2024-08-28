# Import necessary libraries for the Flask application
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import pandas as pd
from pandasai import Agent

# Import specific chat models from their respective libraries
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

# Initialize the Flask application
app = Flask(__name__)

# Load environment variables from a .env file
load_dotenv()

# Define a route for the '/analyst' endpoint that accepts POST requests
@app.route('/analyst', methods=['POST'])
def analyst():
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
        llm = ChatOpenAI(model_name=model_name, temperature=0, seed=26, api_key=os.environ['DEEPSEEK_API_KEY'])
    elif llm_type == 'Mistral':
        llm = ChatMistralAI(model_name=model_name, temperature=0, seed=26, api_key=os.environ['MISTRAL_API_KEY'])
    elif llm_type == 'OpenAI':
        llm = ChatOpenAI(model_name=model_name, temperature=0, api_key=os.environ['OPENAI_API_KEY'])

    # Initialize the agent with the data and configuration
    agent = Agent(data, config={"llm": llm, "open_charts": False})

    # Perform the chat operation and get the response and explanation
    response = agent.chat(chat)
    explanation = agent.explain()

    # Convert the response to a dictionary if it's a DataFrame
    if isinstance(response, pd.DataFrame):
        response_dict = response.to_dict(orient='records')
    elif isinstance(response, list):
        response_dict = [item if isinstance(item, dict) else item.__dict__ for item in response]
    else:
        response_dict = response if isinstance(response, dict) else response.__dict__
    return jsonify({"response": response_dict, "explanation": explanation})

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)
