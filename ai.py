import logging
import os
import re
from dotenv import load_dotenv
from pandasai.llm import BambooLLM
from database_pg import get_user_by_username, log_token_usage
from vector_store import VectorStore

# Import specific chat models from their respective libraries
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


# Import specific embeddings models from their respective libraries
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

load_dotenv()  # Load environment variables from .env file

from typing import List

class CompletionRequest:
    def __init__(self, username:str, info: List[str], chat: List[str]):
        self.username = username
        self.info = info
        self.chat = chat

class CompletionResponse:
    def __init__(self, error: str = None, answer: str = None, vectors: List[dict] = None):
        self.error = error
        self.answer = answer
        self.vectors = vectors

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
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=os.environ['GOOGLE_API_KEY'])
    elif llm_type == 'Mistral':
        return ChatMistralAI(model_name=model, temperature=temperature, seed=seed, api_key=os.environ['MISTRAL_API_KEY'])
    elif llm_type == 'Anthropic':
        model_kwargs = {'seed': seed}
        return ChatAnthropic(model_name=model, temperature=temperature, api_key=os.environ['ANTHROPIC_API_KEY'])
    elif llm_type == 'OpenAI':
        return ChatOpenAI(model_name=model, temperature=1, seed=seed, api_key=os.environ['OPENAI_API_KEY'])
    elif llm_type == 'Ollama':
        return ChatOpenAI(model_name=model, temperature=temperature, base_url='http://192.168.1.9:11434/v1', api_key='ollama')
    elif llm_type == 'Llama.cpp':
        return ChatOpenAI(model_name=model, temperature=temperature, base_url='http://192.168.1.9:8080/v1', api_key='ollama')
    else:
        raise ValueError(f"Unsupported llm_type: {llm_type}")

def choose_emb_model(emb_llm_type, emb_model):
    """
    Choose and initialize the appropriate embeddings model based on the provided type and model.

    :param emb_llm_type: Type of the embeddings model (e.g., 'Mistral', 'Ollama', 'OpenAI')
    :param emb_model: Model name
    :return: Initialized embeddings model instance
    """
    if emb_llm_type == 'Mistral':
        mistralai_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistralai_api_key:
            logging.error("MISTRAL_API_KEY environment variable is not set")
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        return MistralAIEmbeddings(model=emb_model, api_key=mistralai_api_key)
    elif emb_llm_type == 'Ollama':
        return OllamaEmbeddings(model=emb_model,base_url='http://192.168.1.9:11434')
    elif emb_llm_type == 'OpenAI':
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logging.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return OpenAIEmbeddings(model=emb_model, api_key=openai_api_key)
    else:
        logging.error(f"Unsupported emb_llm_type: {emb_llm_type}")
        raise ValueError(f"Unsupported emb_llm_type: {emb_llm_type}")

def complete_chat(req: CompletionRequest, store: VectorStore, llm_type:str, model:str):
    #emb_llm_type = "OpenAI"
    #emb_llm_type = "Ollama"
    #llm_type = "Groq" 
    #model = "llama-3.1-8b-instant"
    #emb_model ="mistral-embed"
    #emb_model ="text-embedding-ada-002"
    #emb_model = "jeffh/intfloat-multilingual-e5-large:f16"
    #emb_model = "bge-m3:latest"

    logging.info(f"Starting chat completion with llm_type: {llm_type}, model: {model}")
    question = req.chat[-1]
    logging.info(f"Processing question: {question}")
    vectors: List[dict] = [];
    try:
        vectors = store.find_similar_vectors(question, 3, 0.5)
    except Exception as e:
        logging.error(str(e))
        return CompletionResponse(error=str(e))
    if not req.info and not vectors:
        logging.info("No information available for the question")
        return CompletionResponse(answer="Non ho informazioni al riguardo")
    logging.info(f"Found {len(vectors)} relevant paragraphs")

    messages = [{"role": "system", "content": """You are Dino, an assistant who helps users by answering questions concisely.
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
6. Only say 'I have no information about this' if the context contains ABSOLUTELY NOTHING relevant to the question"""}]

    # Format context with clear sections and metadata
    context = ""
    #if req.info:
    #    context_parts.append("BACKGROUND INFORMATION:\n-------------------\n" + "\n".join(req.info))
    if vectors:
        context += "RELEVANT CONTEXT:\n----------------"
    for vec in vectors:
        context += "\n" + vec['metadata']['text']
    
    if context:
        messages.append({"role": "user", "content": "Here is the context you MUST use to answer questions:\n\n" + context})
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
        if user:
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
            return CompletionResponse(answer=answer, vectors=vectors)
    except Exception as e:   
        logging.error(f"Error in chat completion: {str(e)}")
        return CompletionResponse(error=f"Error in chat completion: {str(e)}")

def reply_to_prompt(prompt: str, username: str, llm_type: str, model: str) -> str:
    messages = [
        {"role": "system", "content": """Sei un esperto di enti non-profit e devi realizzare il rapporto annuale della tua organizzazione.
Io ti chiederò di scrivere una sezione alla volta, dandoti indicazioni sui contenuti da includere in ciascuna sezione.
Usa un linguaggio preciso ma non troppo tecnico, che sia comprensibile anche al pubblico generale.
Non usare elenchi puntati o numerati. Non inserire titoli. Non aggiungere testo all'inizio o alla fine.
Non aggiungere paragrafi di conclusione o chiusura. Non usare espressioni come "in questo documento" usa invece "in questa sezione".
Scrivi sempre in italiano e genera l'output solo testo senza markdown o html.
Se non hai informazioni sufficienti per rispondere non rispondere niente."""},
        {"role": "user", "content": prompt}
    ]
    llm = choose_llm(llm_type, model, temperature=0.8)
    try:
        resp = llm.invoke(messages)
        # token_usage = resp.response_metadata.get('token_usage', {})
        # token_in = token_usage.get('prompt_tokens', 0)
        # token_out = token_usage.get('completion_tokens', 0)
        # user = get_user_by_username(username)
        # log_token_usage(user_id=user.get("id"), token_input=token_in, token_output=token_out, model=model, provider=llm_type)
        return resp.content
    except Exception as e:
        logging.error(f"Error in prompt completion: {str(e)}")
        raise e

def describe_image(url: str, provider: str, model: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "descrivi brevemente il contenuto di questa immagine"},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    }]
    llm = choose_llm(provider, model, temperature=0.8)
    resp = llm.invoke(messages)
    return resp.content

def audioFormPromptBuild(formSchema, formSchemaExampleData, formSchemaName:str, formSchemaChoices, transcribedAudio:str):
    
    if not formSchema or not formSchemaExampleData or not formSchemaName or not transcribedAudio:
        return

    formSchemaExampleData = (
        "{{'case_name': string, "
        "'dob': date, "
        "'eta': integer, "
        "'migrante': boolean, "
        "'problemi': multichoice, "
        "'rate_visita': range, "
        "'commenti': text, "
        "'trascrizione_integrale': text}}"
    )

    formFieldsDescrition = (
        "{{'case_name': E' il nome della persona di cui stiamo raccogliendo le informazioni "
        "'dob': E' la data di nascita della persone di cui stiamo raccogliendo le informazioni "
        "'eta': E' l'età della persona di cui stiamo raccogliendo le informazioni"
        "'migrante': Indica se la persona è migrante o meno"
        "'problemi': Seleziona tutti i problemi della persona di cui stiamo raccogliendo le informazioni"
        "'rate_visita': Valuta il rating della visita analizzando il campo commenti"
        "'commenti': Sono le informazioni contenute nella registrazione che non riguardano campi strutturati del form"
        "'trascrizione_integrale': text}}"
    )

    system = f"""
    Sei un assistente specializzato nell'estrazione di dati da trascrizioni audio.
    Rispondi ESCLUSIVAMENTE in formato JSON valido.
    Non aggiungere commenti, spiegazioni o testo aggiuntivo.
    """

    user = f"""
    DATI INPUT:
    Schema form: {formSchemaName}
    Opzioni disponibili: {formSchemaChoices}
    Template di output: {formSchemaExampleData}
    Descrizione dei campi: {formFieldsDescrition}
    Trascrizione audio: {transcribedAudio}
    
    ISTRUZIONI:
    Compila il template JSON utilizzando SOLO le informazioni dalla trascrizione.

    REGOLE PER CAMPO:
    - boolean: true/false basato sulla trascrizione
    - multichoice: array di valori da "Opzioni disponibili". Se menzionata opzione non presente e se tra le Opzioni disponibili esiste "altro", includi "altro"
    - singlechoice: array di valori da "Opzioni disponibili". Se menzionata opzione non presente e se tra le Opzioni disponibili esiste "altro", includi "altro"
    - date: formato YYYY-MM-DD
    - text/string: testo estratto dalla trascrizione
    - range: valore numerico
    - number: valore numerico estratto dalla trascrizione
    - Se informazione è mancante tralascia il campo e non mettere null

    OUTPUT: JSON compilato seguendo il template fornito.
    """
    return {'systemprompt': system, 'userprompt': user}

def audioFormCompilation(userprompt: str, systemprompt: str, username:str, llm_type: str, model:str):
    if not userprompt or not systemprompt or not llm_type or not model or not username:
        return
    messages = [
        {"role": "system", "content": systemprompt},
        {"role": "user", "content": userprompt}
    ]
    print(messages)
    llm = choose_llm(llm_type, model, temperature=0)
    
    try:
        resp = llm.invoke(messages)
        token_usage = resp.response_metadata.get('token_usage',{})
        token_in = token_usage.get('prompt_tokens',0)
        token_out = token_usage.get('completion_tokens',0)
        user = get_user_by_username(username)
        log_token_usage(user_id=user.get("id"), token_input=token_in, token_output=token_out, model=model, provider=llm_type)
        #print(resp.content)
        #clean = re.sub(r"<think>.*?</think>\n?", "", resp.content, flags=re.DOTALL).strip()
        #answer = clean if clean else resp.content
        #print(answer)
        #return answer
        return resp.content
    except Exception as e:
        logging.error(f"Error in audio form compilation: {str(e)}")
        return CompletionResponse(error=f"Error in Audio Form Compilation: {str(e)}")
