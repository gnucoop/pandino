# Pandino

## Overview
Pandino is a powerful tool designed to analyze and visualize data using various language models. It provides a flexible API for integrating different LLMs and processing data efficiently.

## Features
- Supports multiple LLM types including Groq, Deepseek, and Mistral.
- Provides a Flask-based API for easy integration and usage.
- Handles data processing and visualization using popular libraries like pandas and plotly.

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

## Usage
To run the Pandino API service, use the following command:
```bash
python main.py
```

To access the `/analyst` endpoint using `curl`, you can use the following command:
```bash
curl -X POST "http://127.0.0.1:5000/analyst" -H "Content-Type: application/json" -d '{"model_name": "llama-3.1-70b-versatile", "llm_type": "Groq", "chat": "Chat with your data...", "data": "file.csv", "config": {"open_charts": false}}'
```

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before getting started.

## License
This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.