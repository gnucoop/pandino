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
   MISTRAL_API_KEY=your_mistral_api_key
   OPENAI_API_KEY=your_openai_api_key
   ENCRYPTION_KEY=your_encryption_key_for_database
   ```

## Usage

### User Management with SQLite
Pandino includes a secure user management system using SQLite. The system allows you to add users and their API keys to the database.

#### Initializing the Database and Adding Users
To initialize the SQLite database and add a new user, use the following command:
```bash
python database.py add_user <username> <api_key>
```

#### Listing Users
To list all users in the database:
```bash
python database.py list_users
```

### Running the Pandino API Service
To run the Pandino API service, use the following command:
```bash
python main.py
```

### Accessing the API
To access the `/analyst` endpoint using `curl`, use the following command:
```bash
curl -X POST "http://127.0.0.1:5000/analyst" \
     -H "Content-Type: application/json" \
     -H "X-API-KEY: your_api_key_here" \
     -d '{
         "model_name": "llama-3.1-70b-versatile",
         "llm_type": "Groq",
         "chat": "Analyze the data and provide insights",
         "data": "path/to/your/data.csv"
     }'
```

Replace `your_api_key_here` with a valid API key from the database, and adjust the `model_name`, `llm_type`, `chat`, and `data` fields as needed.

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before getting started.

## License
This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.
