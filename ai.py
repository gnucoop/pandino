import logging
import os
from dotenv import load_dotenv
import pandas as pd
from pandasai.llm import BambooLLM

# Import specific chat models from their respective libraries
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

#from langchain_ollama import ChatOllama

# Import specific embeddings models from their respective libraries
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

#Import specific vector store database from their specific libraries
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

import database_pg
from database_pg import get_user_by_username, log_token_usage

load_dotenv()  # Load environment variables from .env file

from typing import List

class CompletionRequest:
    def __init__(self, dino_graphql: str, auth_token: str, namespace: str, username:str, info: List[str], chat: List[str]):
        self.dino_graphql = dino_graphql
        self.auth_token = auth_token
        self.namespace = namespace
        self.username = username
        self.info = info
        self.chat = chat

class CompletionResponse:
    def __init__(self, error: str = None, paragraphs: List[str] = None, similarities: List[float] = None, pages: List[str] = None, sources: List[str] = None, urls: List[str] = None, mimetypes: List[str] = None,  answer: str = None):
        self.error = error
        self.paragraphs = paragraphs
        self.similarities = similarities
        self.pages = pages
        self.sources = sources
        self.urls = urls
        self.mimetypes = mimetypes
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
    elif llm_type == 'BambooLLM':
        return BambooLLM(api_key=os.environ['PANDASAI_API_KEY'])
    elif llm_type == 'Together':
        return ChatOpenAI(model_name=model, temperature=temperature, seed=seed, base_url='https://api.together.xyz/v1', api_key=os.environ['TOGETHER_API_KEY'])
    elif llm_type == 'Google':
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, seed=seed, google_api_key=os.environ['GOOGLE_API_KEY'])
    elif llm_type == 'Mistral':
        return ChatMistralAI(model_name=model, temperature=temperature, seed=seed, api_key=os.environ['MISTRAL_API_KEY'])
    elif llm_type == 'Anthropic':
        model_kwargs = {'seed': seed}
        return ChatAnthropic(model_name=model, temperature=temperature, api_key=os.environ['ANTHROPIC_API_KEY'])
    elif llm_type == 'OpenAI':
        return ChatOpenAI(model_name=model, temperature=temperature, seed=seed, api_key=os.environ['OPENAI_API_KEY'])
    elif llm_type == 'Ollama':
        return ChatOpenAI(model_name=model, temperature=temperature, base_url='http://192.168.1.8:11434/v1', api_key='ollama')
    elif llm_type == 'glhf':
        return ChatOpenAI(model_name=model, temperature=temperature, base_url='https://glhf.chat/api/openai/v1', api_key=os.environ['GLHF_API_KEY'])
    elif llm_type == 'Llama.cpp':
        return ChatOpenAI(model_name=model, temperature=temperature, base_url='http://192.168.1.8:8080/v1', api_key='ollama')
    else:
        raise ValueError(f"Unsupported llm_type: {llm_type}")

def complete_chat(req: CompletionRequest, llm_type:str, model:str, emb_llm_type:str, emb_model:str):                                                                                                                                        
     #emb_llm_type = "OpenAI"
     #emb_llm_type = "Ollama"
     #llm_type = "Groq" 
     #model = "llama-3.1-8b-instant"
     #emb_model ="mistral-embed"
     #emb_model ="text-embedding-ada-002"
     #emb_model = "jeffh/intfloat-multilingual-e5-large:f16"
     #emb_model = "bge-m3:latest"

     logging.info(f"Starting chat completion with llm_type: {llm_type}, model: {model}") 
    #  if len(req.chat) % 2 == 0:                                                                                                                                                                               
    #      logging.error("Chat completion error: chat must be a list of user,assistant messages ending with a user message")                                                                                    
    #      return CompletionResponse(error="Chat completion error: chat must be a list of user,assistant messages ending with a user message")                                                                  
     question = req.chat[-1]                                                                                                                                                                                  
     logging.info(f"Processing question: {question}")                                                                                                                                                         
     logging.info(f"Namespace: {req.namespace}")                                                                                                                                                              
     paragraphs, similarities, sources, pages, urls, mimetypes, err = find_similar_paragraphs(question, 3, 0.5, req.namespace, emb_llm_type=emb_llm_type, model=emb_model) 
     if err: 
         logging.error(f"Error finding similar paragraphs: {err}")                                                                                                                                            
         return CompletionResponse(error=f"Error finding similar paragraphs: {err}")                                                                                                                          
     if not req.info and not paragraphs: 
         logging.info("No information available for the question") 
         return CompletionResponse(answer="Non ho informazioni al riguardo") 
     logging.info(f"Found {len(paragraphs)} relevant paragraphs")
     # Start with a very explicit system message about using context
     # Start with a greeting for the first message
     if len(req.chat) == 1:
         return CompletionResponse(answer="Hello! How can I help you?")

     messages = [
         {"role": "system", "content": """You are Dino, an assistant who helps users by answering questions concisely.
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
6. Only say 'I have no information about this' if the context contains ABSOLUTELY NOTHING relevant to the question"""}
     ]

     # Format context with clear sections and metadata
     context_parts = []
     #if req.info:
     #    context_parts.append("BACKGROUND INFORMATION:\n-------------------\n" + "\n".join(req.info))
     if paragraphs:
         context_parts.append("RELEVANT CONTEXT:\n----------------\n" + "\n".join(f"[Similarity: {similarities[i]:.2f}] {p}" for i, p in enumerate(paragraphs)))
     
     if context_parts:
         context_message = "\n\n".join(context_parts)
         messages.append({"role": "user", "content": "Here is the context you MUST use to answer questions:\n\n" + context_message})
         messages.append({"role": "assistant", "content": "I have received the context and will ONLY use this information to answer questions. I will not make up or infer information not present in this context."})

     # Add the chat history if it exists
     if len(req.chat) > 1:
         messages.append({"role": "user", "content": "PREVIUOS CONVERSATION CONTEXT:\n-------------------------"})
         for i in range(0, len(req.chat)-1, 2):
             messages.append({"role": "assistant", "content": f"ASSISTANT: {req.chat[i]}"})
             messages.append({"role": "user", "content": f"USER: {req.chat[i+1]}"})

     # Add the final user question with very explicit instructions
     messages.append({"role": "user", "content": f"""CURRENT QUESTION:
----------------
{req.chat[-1]}

IMPORTANT INSTRUCTIONS:
1. Search through ALL the context provided above
2. Find ANY relevant information that relates to this question
3. If you find ANY relevant information, use it to answer
4. Only say 'I have no information about this' if you find ABSOLUTELY NOTHING relevant
5. Your answer must ONLY use information from the provided context"""})
     llm = choose_llm(llm_type, model)

     try: 
         resp = llm.invoke(messages) 
         token_usage = resp.response_metadata.get('token_usage',{})
         token_in = token_usage.get('prompt_tokens',0)
         token_out = token_usage.get('completion_tokens',0)
         user = get_user_by_username(req.username)
         if user is None: 
            logging.error(f"Chat completion error: could not find any user with this username: {req.username}") 
            return CompletionResponse(error=f"Chat completion error: could not find any user with this username: {req.username}")    
         log_token_usage(user_id=user.get("id"), token_input=token_in, token_output=token_out, model=model, provider=llm_type)
         
         # Check if response indicates no information before returning
         answer = resp.content
         no_info_phrases = [
             "Non ho informazioni",
             "I have no information",
             "I don't have any information",
             "No information available"
         ]
         is_no_info = any(phrase.lower() in answer.lower() for phrase in no_info_phrases)
         
         # Only include paragraphs and metadata if it's not a "no information" response
         if is_no_info:
             return CompletionResponse(answer=answer)
         else:
             return CompletionResponse(answer=answer,
                                     paragraphs=paragraphs,
                                     similarities=similarities,
                                     pages=pages,
                                     sources=sources,
                                     urls=urls,
                                     mimetypes=mimetypes)
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
        #index = connect_to_pinecone("langchain-test-index")
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
        pages = []
        sources = []
        urls = []
        mimetypes = []
        if hasattr(resp, 'matches'):
                for vec in resp.matches:
                    if vec.score >= min_similarity:
                        paragraphs.append(vec.metadata["text"])
                        similarities.append(vec.score)
                        pages.append(vec.metadata["page"])
                        sources.append(vec.metadata["source"])
                        urls.append(vec.metadata["url"])
                        mimetypes.append(vec.metadata["mimetype"])
        else:
            logging.warning("The response from the vector database does not contain 'matches' attribute.")
        
        logging.info(f"Filtered to {len(paragraphs)} paragraphs above minimum similarity")
        return paragraphs, similarities, sources, pages, urls, mimetypes, None
    except Exception as e:
        logging.error(f"Error in find_similar_paragraphs: {str(e)}")
        return [], [], str(e)

def reply_to_prompt(prompt, username:str, llm_type: str, model:str):
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
        user = get_user_by_username(username)
        log_token_usage(user_id=user.get("id"), token_input=token_in, token_output=token_out, model=model, provider=llm_type)
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
        return OllamaEmbeddings(model=model,base_url='http://192.168.1.8:11434')
    elif emb_llm_type == 'OpenAI':
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logging.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return OpenAIEmbeddings(model=model, api_key=openai_api_key)
    else:
        logging.error(f"Unsupported emb_llm_type: {emb_llm_type}")
        raise ValueError(f"Unsupported emb_llm_type: {emb_llm_type}")
