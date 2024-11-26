# Pandino

## Overview
Pandino is a powerful tool designed to analyze and visualize data using various language models. It provides a flexible API for integrating different LLMs and processing data efficiently.

## Features
- Supports multiple LLM types including Groq, Deepseek, Mistral, and OpenAI.
- Provides a Flask-based API for easy integration and usage.
- Handles data processing and visualization using popular libraries like pandas and pandasai.
- Implements a secure user management system using SQLite for API key validation.
- Uses environment variables for secure storage of API keys and encryption keys.

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

3. Set up environment variables:
   Create a `.env` file in the project root and add the following variables:
   ```
   GROQ_API_KEY=your_groq_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   DEEPINFRA_API_KEY=your_deepinfra_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   OPENAI_API_KEY=your_openai_api_key
   ENCRYPTION_KEY=your_encryption_key_for_database
   ```

## Troubleshooting
If you encounter any issues during installation or usage, please check the following:

- Ensure all environment variables are correctly set in the `.env` file.
- Verify that all required dependencies are installed by running `pip install -r requirements.txt` again.
- If you encounter issues with API keys, ensure they are valid and not expired.

## Usage

### User Management with SQLite or Postgres (Use database\_sqlite.py or database\_pg.py accordingly with your choice)
Pandino includes a secure user management system using SQLite and Postgres. The system allows you to add users and their API keys to the database.

#### Initializing the Database
To initialize the SQLite database, use the following command:
```bash
python database_sqlite.py init_db
```

#### Adding Users
To add a new user to the database:
```bash
python database_sqlite.py add_user <username> <api_key>
```

#### Listing Users
To list all users in the database:
```bash
python database_sqlite.py list_users
```

#### Viewing Stored API Keys
To view all stored (encrypted) API keys:
```bash
python database_sqlite.py print_keys
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
     -F "file=your_local_csv"
     
# curl -X POST "http://127.0.0.1:5000/startdatachat" \
#      -H "Content-Type: multipart/form-data" \
#      -H "X-API-KEY: your_api_key_here" \
#      -H "X-USER-NAME: your_full_user_name_here" \
#      -d '{
#          "model_name": "llama-3.1-70b-versatile",
#          "llm_type": "Groq",
#          "file": your_csv
#      }'
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
     -d '{
         "chat": your_request_to_pandas_here
     }'
```

Replace `your_api_key_here` with a valid API key from the database, `your_full_user_name_here` with a user name (it will be used to create the agent dedicated export folder), `your_request_to_pandas_here` with your natural language request to Pandas and adjust the `model_name`, `llm_type`, and `data` fields as needed.

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before getting started. To report bugs or suggest features, please open an issue on the GitHub repository.

## License
This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.
