import logging
import os
from dotenv import load_dotenv

# Import specific chat models from their respective libraries
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
#from langchain_ollama import ChatOllama

# Import specific embeddings models from their respective libraries
#from langchain_mistralai import MistralAIEmbeddings
#from langchain_openai import OpenAIEmbeddings
#from langchain_ollama import OllamaEmbeddings

#Import specific vector store database from their specific libraries
#from pinecone import Pinecone, ServerlessSpec
#from langchain_pinecone import PineconeVectorStore
#from langchain_postgres import PGVector

#import database
#from database import log_token_usage

load_dotenv()  # Load environment variables from .env file

from typing import List

def choose_llm(llm_type, model, temperature=0, seed=26, base_url=None, api_key=None):
    """
    Choose and initialize the appropriate LLM based on the provided type and model.

    :param llm_type: Type of the LLM (e.g., 'Groq', 'Deepseek', 'Mistral', 'OpenAI', 'Ollama')
    :param model: Model name
    :param temperature: Temperature for the model
    :param seed: Seed for the model
    :param base_url: Base URL for the model (if applicable)
    :param api_key: API key for the model
    :return: Initialized LLM instance
    """
    print(llm_type)
    if llm_type == 'Groq':
        model_kwargs = {'seed': seed}
        return ChatGroq(model_name=model, temperature=temperature, api_key=os.environ['GROQ_API_KEY'], model_kwargs=model_kwargs)
    elif llm_type == 'Deepseek':
        return ChatOpenAI(model_name=model, temperature=temperature, seed=seed, base_url='https://api.deepseek.com', api_key=os.environ['DEEPSEEK_API_KEY'])
    elif llm_type == 'Deepinfra':
        return ChatOpenAI(model_name=model, temperature=temperature, seed=seed, base_url='https://api.deepinfra.com/v1/openai', api_key=os.environ['DEEPINFRA_API_KEY'])
    elif llm_type == 'Together':
        return ChatOpenAI(model_name=model, temperature=temperature, seed=seed, base_url='https://api.together.xyz/v1', api_key=os.environ['TOGETHER_API_KEY'])
    elif llm_type == 'Mistral':
        return ChatMistralAI(model_name=model, temperature=temperature, seed=seed, api_key=os.environ['MISTRAL_API_KEY'])
    elif llm_type == 'OpenAI':
        return ChatOpenAI(model_name=model, temperature=temperature, seed=seed, api_key=os.environ['OPENAI_API_KEY'])
    elif llm_type == 'Ollama':
        return ChatOpenAI(model_name=model, temperature=temperature, base_url='http://127.0.0.1:11434/v1', api_key='ollama')
    else:
        raise ValueError(f"Unsupported llm_type: {llm_type}")
