import logging
import os
from dotenv import load_dotenv

# Import specific chat models from their respective libraries
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
#from langchain_ollama import ChatOllama

# Import specific embeddings models from their respective libraries
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

#Import specific vector store database from their specific libraries
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_postgres import PGVector

import databasepg
from databasepg import log_token_usage

load_dotenv()  # Load environment variables from .env file

from typing import List

class CompletionRequest:
    def __init__(self, dino_graphql: str, auth_token: str, namespace: str, info: List[str], chat: List[str]):
        self.dino_graphql = dino_graphql
        self.auth_token = auth_token
        self.namespace = namespace
        self.info = info
        self.chat = chat

class CompletionResponse:
    def __init__(self, error: str = None, paragraphs: List[str] = None, similarities: List[float] = None, answer: str = None):
        self.error = error
        self.paragraphs = paragraphs
        self.similarities = similarities
        self.answer = answer

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

def complete_chat(req: CompletionRequest, llm_type=None, model=None):                                                                                                                                        
     emb_llm_type = "OpenAI"
     #emb_llm_type = "Ollama"
     llm_type = "Groq" 
     model = "llama-3.1-8b-instant"
     #emb_model ="mistral-embed"
     emb_model ="text-embedding-ada-002"
     #emb_model = "jeffh/intfloat-multilingual-e5-large:f16"
     #emb_model = "bge-m3:latest"

     logging.info(f"Starting chat completion with llm_type: {llm_type}, model: {model}")                                                                                                                      
     if len(req.chat) % 2 == 0:                                                                                                                                                                               
         logging.error("Chat completion error: chat must be a list of user,assistant messages ending with a user message")                                                                                    
         return CompletionResponse(error="Chat completion error: chat must be a list of user,assistant messages ending with a user message")                                                                  
     question = req.chat[-1]                                                                                                                                                                                  
     logging.info(f"Processing question: {question}")                                                                                                                                                         
     logging.info(f"Namespace: {req.namespace}")                                                                                                                                                              
     paragraphs, similarities, err = find_similar_paragraphs(question, 2, 0.7, req.namespace, emb_llm_type=emb_llm_type, model=emb_model)                                                                             
     if err:                                                                                                                                                                                                  
         logging.error(f"Error finding similar paragraphs: {err}")                                                                                                                                            
         return CompletionResponse(error=f"Error finding similar paragraphs: {err}")                                                                                                                          
     if not req.info and not paragraphs:                                                                                                                                                                      
         logging.info("No information available for the question")                                                                                                                                            
         return CompletionResponse(answer="Non ho informazioni al riguardo")                                                                                                                                  
     logging.info(f"Found {len(paragraphs)} relevant paragraphs")                                                                                                                                             
                                                                                                                                                                                                              
     messages = [                                                                                                                                                                                             
         {"role": "system", "content": "You are Dino, an assistant who helps users by answering questions concisely."},                                                                          
         {"role": "user", "content": "In this chat, I will send you various information, followed by questions. Answer the questions with the information contained in this chat. If the answer is not contained in the information, answer 'I have no information about this'."},                                                                                                                                 
         {"role": "assistant", "content": "Ok!"},                                                                                                                                                             
     ]                                                                                                                                                                                                        
     for info in req.info:                                                                                                                                                                                    
         messages.append({"role": "user", "content": "Information:\n" + info})                                                                                                                               
     for info in paragraphs:                                                                                                                                                                                  
         messages.append({"role": "user", "content": "Information:\n" + info})                                                                                                                               
     messages.append({"role": "assistant", "content": "Information received!"})                                                                                                                              
     for i, msg in enumerate(req.chat):                                                                                                                                                                       
         role = "user" if i % 2 == 0 else "assistant"                                                                                                                                                         
         messages.append({"role": role, "content": msg})                                                                                                                                                      
                                                                                                                                                                                                              
     llm = choose_llm(llm_type, model)

     try:                                                                                                                                                                                                     
         resp = llm.invoke(messages)                                                                                                                                                                          
         token_usage = resp.response_metadata.get('token_usage',{})
         token_in = token_usage.get('prompt_tokens',0)
         token_out = token_usage.get('completion_tokens',0)
         user_id = 2
         log_token_usage(user_id=user_id, token_input=token_in, token_output=token_out, model=model, provider=llm_type)
         return CompletionResponse(answer=resp.content,
                                   paragraphs=paragraphs,
                                   similarities=similarities)                                                                                                                                                       
     except Exception as e:                                                                                                                                                                                   
         logging.error(f"Error in chat completion: {str(e)}")                                                                                                                                                 
         return CompletionResponse(error=f"Error in chat completion: {str(e)}")

def embed(emb_llm_type, model, text):
    logging.info(f"Attempting to embed text with {emb_llm_type} model: {model}")
    logging.debug(f"Text to embed (first 50 chars): {text[:50]}...")

    embeddings = choose_emb_model(emb_llm_type, model)

    try:
        logging.info("Attempting to embed query")
        single_vector = embeddings.embed_query(text)
        logging.info(f"Successfully embedded text with {emb_llm_type}")
        logging.debug(f"Embedded vector (first 5 elements): {single_vector[:5]}")
        return single_vector
    except Exception as e:
        logging.error(f"Error embedding text with {emb_llm_type}: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response'):
            logging.error(f"Response content: {e.response.content}")
        if hasattr(e, '__dict__'):
            logging.error(f"Error attributes: {e.__dict__}")
        raise ValueError(f"Error embedding text with {emb_llm_type}: {str(e)}")

def connect_to_pinecone (index_name: str):
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    return index

def connect_to_vector_db (db_name: str, index_name: str, emb_llm_type: str, emb_model):
    if db_name == 'pinecone':
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        embeddings = choose_emb_model (emb_llm_type, emb_model)
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        logging.info("VectorStore created")
        return vector_store
    elif db_name == 'pgvector':
        embeddings = choose_emb_model (emb_llm_type, emb_model)
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
        vector_store = PGVector(embeddings=embeddings,collection_name=index_name,connection=connection,use_jsonb=True)
        return vector_store

def find_similar_paragraphs(text: str, top_k: int, min_similarity: float, namespace: str, emb_llm_type: str, model: str) -> tuple:
    logging.info(f"Finding similar paragraphs for text: {text[:50]}...")
    
    try:
        
        """
        db_name='pinecone'
        index_name='index'
        emb_llm_type ='OpenAI'
        emb_model='text-embedding-ada-002'

        vector_store = connect_to_vector_db(db_name=db_name,index_name=index_name,emb_llm_type=emb_llm_type,emb_model=emb_model)
        print(vector_store)
        logging.info("Connected to Vector Database")
        
        try:
            resp = vector_store.similarity_search_with_relevance_scores (text, k=top_k,score_threshold=min_similarity)
            print(resp)
            for res, score in resp:
                print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
        except Exception as e:
            logging.error(f"Error querying the vector database: {str(e)}")
            return [], [], str(e)
            """
        
        vec = embed(emb_llm_type, model, text)
        logging.info("Text embedded successfully")
        index = connect_to_pinecone("index")
        resp = index.query(
            vector=vec, 
            top_k=top_k, 
            include_metadata=True, 
            namespace=namespace,
            min_similarity=min_similarity
        )
        
        logging.info(f"Vector Database query completed, found {len(resp.matches)} matches")
        
        paragraphs = []
        similarities = []
        if hasattr(resp, 'matches'):
                for vec in resp.matches:
                    if vec.score >= min_similarity:
                        paragraphs.append(vec.metadata["text"])
                        similarities.append(vec.score)
        else:
            logging.warning("The response from the vector database does not contain 'matches' attribute.")
        
        logging.info(f"Filtered to {len(paragraphs)} paragraphs above minimum similarity")
        return paragraphs, similarities, None
    except Exception as e:
        logging.error(f"Error in find_similar_paragraphs: {str(e)}")
        return [], [], str(e)

def reply_to_prompt(prompt):
    llm_type = "Ollama"
    model = "gemma2:2b"
    messages = [
        {"role": "system", "content": "Sei un esperto di monitoraggio e valutazione che supporta le Organizzazioni non governative a scrivere il proprio bilancio sociale."},
        {"role": "user", "content": prompt}
    ]

    llm = choose_llm(llm_type, model, temperature=0.3)

    try:
        resp = llm.invoke(messages)
        token_usage = resp.response_metadata.get('token_usage',{})
        token_in = token_usage.get('prompt_tokens',0)
        token_out = token_usage.get('completion_tokens',0)
        user_id = 2
        log_token_usage(user_id=user_id, token_input=token_in, token_output=token_out, model=model, provider=llm_type)
        return CompletionResponse(answer=resp.content)
    except Exception as e:
        logging.error(f"Error in chat completion: {str(e)}")
        return CompletionResponse(error=f"Error in chat completion: {str(e)}")
def choose_emb_model(emb_llm_type, model):
    """
    Choose and initialize the appropriate embeddings model based on the provided type and model.

    :param emb_llm_type: Type of the embeddings model (e.g., 'Mistral', 'Ollama', 'OpenAI')
    :param model: Model name
    :return: Initialized embeddings model instance
    """
    if emb_llm_type == 'Mistral':
        mistralai_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistralai_api_key:
            logging.error("MISTRAL_API_KEY environment variable is not set")
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        return MistralAIEmbeddings(model=model, api_key=mistralai_api_key)
    elif emb_llm_type == 'Ollama':
        return OllamaEmbeddings(model=model)
    elif emb_llm_type == 'OpenAI':
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logging.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return OpenAIEmbeddings(model=model, api_key=openai_api_key)
    else:
        logging.error(f"Unsupported emb_llm_type: {emb_llm_type}")
        raise ValueError(f"Unsupported emb_llm_type: {emb_llm_type}")
