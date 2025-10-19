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
  - **Audio Transcription**: Transcribe audio files into text using Whisper.
  - **Image Analysis**: Generate descriptions for images.
  - **Audio-to-Form**: Automatically fill out structured forms from spoken input.

- **Robust Administration and Management**:
  - **Admin Dashboard**: A web interface to monitor application stats, manage users, and view logs.
  - **User & Token Management**: Securely manage users, API keys, and token balances.
  - **Dynamic Prompt Management**: Create, edit, and version control prompts directly from the admin panel without deploying new code.

- **Secure and Configurable**:
  - **Environment-based Configuration**: Easily configure API keys, database connections, and model choices using a `.env` file.
  - **Encrypted API Keys**: User API keys are encrypted in the database for enhanced security.

## Installation
To install Pandino, follow these steps:

1. Clone the repository:
   ```bash
   git clone git@github.com:tulas75/pandino.git
   cd pandino
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup admin panel hash
   ```python
   import bcrypt

   password = b"my-strong-password"
   hashed = bcrypt.hashpw(password, bcrypt.gensalt())

   print(hashed.decode()) 
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add the variables from the `.env.example` file.

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
- **POST /transcribe**: Transcribes an audio file using Whisper.
- **POST /audioformcompilation**: Compiles a form from transcribed audio.
- **GET /health**: Returns the health status of the application.

## Usage

### User Management with PostgreSQL
Pandino includes a secure user management system using PostgreSQL. The system allows you to add users and their API keys to the database.

#### Initializing the Database
To initialize the PostgreSQL database, use the following command:
```bash
python database_pg.py init_db
```

#### Adding Users
To add a new user to the database:
```bash
python database_pg.py add_user <username> <api_key>
```

#### Listing Users
To list all users in the database:
```bash
python database_pg.py list_users
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

Replace `your_api_key_here` with a valid API key from the database, `your_full_user_name_here` with a user name (it will be used to create the agent dedicated export folder), `your_request_to_pandas_here` with your natural language request to Pandas and adjust the `model_name`, `llm_type`, and `data` fields as needed.

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before getting started. To report bugs or suggest features, please open an issue on the GitHub repository.

## License
This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.