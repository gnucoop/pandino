# Pandino

## Overview
Pandino is a powerful tool designed to analyze and visualize data using various language models. It provides a flexible API for integrating different LLMs and processing data efficiently.

## Features
- Supports multiple LLM types including Groq, Deepseek, Mistral, and OpenAI.
- Provides a Flask-based API for easy integration and usage.
- Handles data processing and visualization using popular libraries like pandas and pandasai.
- Implements a secure user management system using SQLite for API key validation.
- Uses environment variables for secure storage of API keys and encryption keys.

## New Features
- **Feedback Submission**: Users can submit feedback via the `/feedback` endpoint.
- **Prompt Handling**: Users can submit prompts via the `/prompt.txt` endpoint.
- **Completion Handling**: Users can request chat completions via the `/completion.json` endpoint.
- **Data Chat**: Users can interact with data via the `/datachat` endpoint.
- **Start Data Chat**: Users can start a data chat session via the `/startdatachat` endpoint.
- **End Data Chat**: Users can end a data chat session via the `/enddatachat` endpoint.
- **API Key Validation**: Users can validate their API keys via the `/validateapikey` endpoint.

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

## Troubleshooting
If you encounter any issues during installation or usage, please check the following:

- Ensure all environment variables are correctly set in the `.env` file.
- Verify that all required dependencies are installed by running `pip install -r requirements.txt` again.
- If you encounter issues with API keys, ensure they are valid and not expired.

## Usage

### User Management with SQLite/Postgresql
Pandino includes a secure user management system using SQLite/Postgresql. The system allows you to add users and their API keys to the database.

Use database_pg.py for Postgresql
Use database.py for SQLite


#### Initializing the Database
To initialize the SQLite database, use the following command:
```bash
python database.py init_db
```

#### Adding Users
To add a new user to the database:
```bash
python database.py add_user <username> <api_key>
```

#### Listing Users
To list all users in the database:
```bash
python database.py list_users
```

#### Viewing Stored API Keys
To view all stored (encrypted) API keys:
```bash
python database.py print_keys
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

## Function Descriptions

### `welcome()`
- **Description**: Returns a welcome message for the root endpoint.
- **Endpoint**: `/`
- **Method**: `GET`
- **Returns**: A welcome message.

### `validate()`
- **Description**: Validates the API key provided in the request headers.
- **Endpoint**: `/validateapikey`
- **Method**: `POST`
- **Parameters**: 
  - `X-API-KEY`: The API key to validate.
- **Returns**: A JSON response indicating whether the API key is valid.

### `endChat()`
- **Description**: Ends the data chat session for the given API key and user name.
- **Endpoint**: `/enddatachat`
- **Method**: `POST`
- **Parameters**:
  - `X-API-KEY`: The API key.
  - `X-USER-NAME`: The user name.
- **Returns**: A JSON response indicating the status of the operation.

### `startChat()`
- **Description**: Starts a data chat session with the provided parameters.
- **Endpoint**: `/startdatachat`
- **Method**: `POST`
- **Parameters**:
  - `X-API-KEY`: The API key.
  - `X-USER-NAME`: The user name.
  - `model_name`: The model name.
  - `llm_type`: The LLM type.
  - `file`: The CSV file to process.
  - `lang`: The language for the chat.
- **Returns**: A JSON response containing the conversation ID and suggested questions.

### `dataChat()`
- **Description**: Handles data chat requests.
- **Endpoint**: `/datachat`
- **Method**: `POST`
- **Parameters**:
  - `X-API-KEY`: The API key.
  - `chat`: The chat request.
- **Returns**: A JSON response containing the chat response and explanation.

### `completion_handler()`
- **Description**: Handles chat completion requests.
- **Endpoint**: `/completion.json`
- **Method**: `POST`
- **Parameters**:
  - `dinoGraphql`: The Dino GraphQL URL.
  - `authToken`: The authentication token.
  - `chat`: The chat request.
  - `username`: The user name.
- **Returns**: A JSON response containing the chat completion response.

### `prompt_handler()`
- **Description**: Handles prompt requests.
- **Endpoint**: `/prompt.txt`
- **Method**: `POST`
- **Parameters**:
  - `graphqlUrl`: The GraphQL URL.
  - `authToken`: The authentication token.
  - `prompt`: The prompt.
  - `username`: The user name.
- **Returns**: A text response containing the prompt reply.

### `submit_feedback()`
- **Description**: Submits feedback for the given user and endpoint.
- **Endpoint**: `/feedback`
- **Method**: `POST`
- **Parameters**:
  - `X-API-KEY`: The API key.
  - `X-USER-NAME`: The user name.
  - `X-Endpoint`: The endpoint for which feedback is submitted.
  - `feedback`: The feedback text.
- **Returns**: A JSON response indicating the status of the feedback submission.

### `summarize()`
- **Description**: Placeholder for the summarize endpoint.
- **Endpoint**: `/summarize`
- **Method**: `GET`
- **Returns**: A message indicating that the endpoint is not yet implemented.

### `categorize()`
- **Description**: Placeholder for the categorize endpoint.
- **Endpoint**: `/categorize`
- **Method**: `GET`
- **Returns**: A message indicating that the endpoint is not yet implemented.

### `img_comparison()`
- **Description**: Placeholder for the image comparison endpoint.
- **Endpoint**: `/img-comparison`
- **Method**: `GET`
- **Returns**: A message indicating that the endpoint is not yet implemented.

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before getting started. To report bugs or suggest features, please open an issue on the GitHub repository.

## License
This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.
