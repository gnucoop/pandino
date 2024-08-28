# Pandino

## Overview
Pandino is a powerful tool designed to analyze and visualize data using various language models. It provides a flexible API for integrating different LLMs and processing data efficiently.

## Features
- Supports multiple LLM types including Groq, Deepseek, and Mistral.
- Provides a Flask-based API for easy integration and usage.
- Handles data processing and visualization using popular libraries like pandas and plotly.
- Implements a basic user management system using SQLite for API key validation.

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

   Note: `sqlite3` is a standard library module in Python and does not need to be installed separately.

## Usage

### API Key Authentication
To ensure secure access to the `/analyst` endpoint, Pandino uses API key authentication. You need to include the `X-API-KEY` header in your requests with a valid API key. The API key should be set in your `.env` file as `API_KEY`.

### User Management with SQLite
Pandino includes a basic user management system using SQLite. The system allows you to add users and their API keys to the database. The `database.py` file contains functions to initialize the database, add users, and validate API keys.

#### Initializing the Database
To initialize the SQLite database, run the following command:
```bash
python database.py
```

#### Adding Users and API Keys
To add a new user and API key, you can use the following command:
```bash
python database.py add_user <username> <api_key>
```

#### API Key Validation
The API key validation is handled by the `validate_api_key` function in the `database.py` file. This function checks if the provided API key exists in the database and returns `True` if it does, otherwise `False`.
To run the Pandino API service, use the following command:
```bash
python main.py
```

To access the `/analyst` endpoint using `curl`, you can use the following command:
```bash
curl -X POST "http://127.0.0.1:5000/analyst" -H "Content-Type: application/json" -H "X-API-KEY: your_secret_api_key_here" -d '{"model_name": "llama-3.1-70b-versatile", "llm_type": "Groq", "chat": "Chat with your data...", "data": "file.csv", "config": {"open_charts": false}}'
```

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before getting started.

## License
This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.
