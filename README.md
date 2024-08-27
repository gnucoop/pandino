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

You can then access the `/analyst` endpoint with the required parameters. For example:
```
http://127.0.0.1:5000/analyst?model_name=mistral-model&llm_type=Mistral&chat=Print%20a%20table%20of%20Top%2010%20Organisation%20aggregate%20by%20tot_dip&data=compass.csv
```

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before getting started.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
