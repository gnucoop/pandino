# Pandino

## Overview
Pandino is a powerful tool designed to analyze and visualize data using various language models. It provides a flexible API for integrating different LLMs and processing data efficiently.

## Features

- **Flexible LLM Integration**: Dynamically choose from a wide range of language models from various providers, including:
  - Groq
  - OpenAI (including Deepseek and Deepinfra)
  - Google
  - Anthropic
  - Mistral
  - Together
  - Local models via Ollama and Llama.cpp

- **Conversational Data Analysis**: Upload CSV files and interact with your data using natural language queries, powered by PandasAI.

- **Retrieval-Augmented Generation (RAG)**: 
  - **Multi-format File Processing**: Ingest and process various file formats (PDF, TXT, Markdown, audio) to build a knowledge base.
  - **Vector Database Support**: Choose between Pinecone and PGVector for efficient similarity search.
  - **Context-aware Completions**: Enhance LLM responses with relevant information retrieved from your documents.

- **Rich API Services**:
  - **RESTful Endpoints**: A comprehensive Flask-based API provides access to all features.
  - **Audio Transcription**: Transcribe audio files into text using the configured ASR provider.
  - **Image Analysis**: Generate descriptions for images.
  - **Audio-to-Form**: Automatically fill out structured forms from spoken input.
  - **Document Comparison**: Compare two documents using OCR and structured extraction to surface their differences.

- **Robust Administration and Management**:
  - **Admin Dashboard**: A web interface to monitor application stats, manage users, and view logs.
  - **User & Token Management**: Securely manage users, API keys, and token balances.
  - **Dynamic Prompt Management**: Create, edit, and version control prompts directly from the admin panel without deploying new code.

- **Secure and Configurable**:
  - **Environment-based Configuration**: Easily configure API keys, database connections, and model choices using a `.env` file.
  - **Encrypted API Keys**: User API keys are encrypted in the database for enhanced security.

## Installation

Pandino targets **Python 3.10** (the repository pins `3.10.13`). To install Pandino, follow these steps:

1. Clone the repository:
   ```bash
   git clone git@github.com:gnucoop/pandino.git
   cd pandino
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate an encryption key:
   `ENCRYPTION_KEY` is used to encrypt user API keys at rest (via Fernet), so it must be a
   valid Fernet key. Generate one with:
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```
   Copy the output into the `ENCRYPTION_KEY` variable of your `.env` file. Keep this value
   secret and stable — rotating it makes previously stored API keys undecryptable.

4. Setup admin panel hash
   ```python
   import bcrypt

   password = b"my-strong-password"
   hashed = bcrypt.hashpw(password, bcrypt.gensalt())

   print(hashed.decode()) 
   ```
   Copy the output into the `ADMIN_PASSWORD_HASH` variable of your `.env` file.

5. Set up environment variables:
   Create a `.env` file in the project root and add the variables from the `.env.example` file.
   Notable groups include database credentials, model/provider selection, API keys, token
   costs and RAG retrieval parameters. For the vector store you can use **PGVector** (backed by
   your PostgreSQL database) or **Pinecone** (set `PINECONE_API_KEY` and the `RAG_NAMESPACE_*`
   variables). Local models are supported via Ollama (`OLLAMA_BASE_URL`).

### Running with Docker

A `Dockerfile` is provided. To build and run the service in a container:
```bash
docker build -t pandino .
docker run --rm -p 5000:5000 --env-file .env pandino
```

## Available Endpoints

Here is a list of the available non-admin endpoints:

- **GET /**: Returns a welcome message.
- **POST /edittokens**: Edits the number of tokens for a user. Requires a Stripe key for authentication.
- **POST /getusertokens**: Retrieves the number of tokens for a user.
- **POST /checkpandinouser**: Checks if a user exists in the Pandino database and adds them if they don't.
- **POST /validateapikey**: Validates a user's API key.
- **POST /enddatachat**: Ends a data chat session and deletes the agent.
- **POST /startdatachat**: Starts a new data chat session and creates an agent.
- **POST /datachat**: Handles the chat interaction with the data agent.
- **POST /buyreport**: Allows a user to "buy" a report using their tokens.
- **POST /completion.json**: Provides a chat completion service.
- **POST /prompt.txt**: Handles a prompt and returns a response.
- **POST /storeragfile**: Stores a file for Retrieval-Augmented Generation (RAG).
- **POST /transcribe**: Transcribes audio with the configured ASR provider, or extracts text from supported documents/images.
- **POST /audioformcompilation**: Compiles a form from transcribed audio.
- **POST /compare_docs**: Compares two documents (with OCR and structured extraction) and returns their differences.
- **POST /agentchat**: AI agent endpoint powered by Smolagents that uses retrieval tools to answer questions based on stored documents in a specified namespace.
- **POST /feedback**: Submits user feedback, viewable from the admin dashboard.
- **GET /health**: Returns the health status of the application.

The following endpoints are registered but **not yet implemented** (they currently return `501`): `GET /summarize`, `GET /categorize`, `GET /img-comparison`.

## Admin Dashboard

Pandino ships with a web-based admin dashboard, available at `/admin` (e.g. `http://127.0.0.1:5000/admin`).
It is protected by a username/password login — set `ADMIN_USERNAME` and `ADMIN_PASSWORD_HASH`
(see the bcrypt step in [Installation](#installation)) in your `.env` file.

From the dashboard you can:
- View application stats and runtime logs
- Manage users and their token balances
- Create, edit, version and delete prompts without redeploying
- Upload and manage RAG files
- Configure token costs
- Review submitted user feedback

## Usage

### User Management with PostgreSQL
Pandino includes a secure user management system using PostgreSQL. The system allows you to add users and their API keys to the database.

#### Initializing the Database
To initialize the PostgreSQL database, use the following command:
```bash
python infrastructure/database_pg.py init_db
```

#### Adding Users
To add a new user to the database:
```bash
python infrastructure/database_pg.py add_user <username> <api_key>
```

#### Listing Users
To list all users in the database:
```bash
python infrastructure/database_pg.py list_users
```

#### Removing Users
To remove a user from the database:
```bash
python infrastructure/database_pg.py remove_user <username>
```

### Running the Pandino API Service
To run the Pandino API service, use the following command:
```bash
python main.py
```

### Accessing the API
To access the `/startdatachat` endpoint using `curl`, use the following command:
```bash
curl -X POST "http://127.0.0.1:5000/startdatachat" \
     -H "Content-Type: multipart/form-data" \
     -H "X-API-KEY: your_api_key_here" \
     -H "X-USER-NAME: your_full_user_name_here" \
     -F "model_name=llama-3.1-70b-versatile"
     -F "llm_type=Groq"
     -F "file=@your_local_csv.csv"
```

To access the `/enddatachat` endpoint using `curl`, use the following command:
```bash
curl -X POST "http://127.0.0.1:5000/enddatachat" \
     -H "Content-Type: application/json" \
     -H "X-API-KEY: your_api_key_here" \
     -H "X-USER-NAME: your_full_user_name_here" \
     -d '{}'
```

To access the `/datachat` endpoint using `curl`, use the following command:
```bash
curl -X POST "http://127.0.0.1:5000/datachat" \
     -H "Content-Type: application/json" \
     -H "X-API-KEY: your_api_key_here" \
     -d 
     {
         "chat": "your_request_to_pandas_here"
     }
```

To access the `/agentchat` endpoint using `curl`, use the following command:
```bash
curl -X POST "http://127.0.0.1:5000/agentchat" \
     -H "Content-Type: application/json" \
     -H "X-API-KEY: your_api_key_here" \
     -d '{
         "chat": ["What is the main topic of the training material?"],
         "username": "user@example.com",
         "namespace": "Dino",
         "language": "ITA",
         "k": 3,
         "min_similarity": 0.5
     }'
```

Replace `your_api_key_here` with a valid API key from the database, `your_full_user_name_here` with a user name (it will be used to create the agent dedicated export folder), `your_request_to_pandas_here` with your natural language request to Pandas and adjust the `model_name`, `llm_type`, and `data` fields as needed.

## Contributing
Contributions are welcome! To report bugs or suggest features, please open an issue on the GitHub repository.

## License
This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.
