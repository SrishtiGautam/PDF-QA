# PDF-QA


# PDF Query and Slack Integration

This project is a Python-based tool that processes PDF documents, extracts relevant information, and posts responses to a specified Slack channel using OpenAI's language models. It employs natural language processing techniques to answer user queries based on the contents of the PDF.

## Demo Video

![Demo Video](media/PDF-QA-v2.mp4)

## Features

- **PDF Processing**: Extracts text from PDF documents and splits it into manageable chunks.
- **Natural Language Queries**: Users can ask questions related to the content of the PDF.
- **OpenAI Integration**: Utilizes OpenAI's models for generating responses and embeddings.
- **Confidence Handling**: Implements logic to handle low-confidence responses.
- **Exact Match Response**: Returns exact matches from the PDF when queries match exactly, using greedy strategy of token generation.
- **Slack Notifications**: Posts responses directly to a specified Slack channel.
- **Error Handling and Logging**: Includes robust error handling, retry logic, and detailed logging.

## Requirements

- Python 3.x
- Libraries:
  - `openai`
  - `slack_sdk`
  - `sklearn`
  - `PyPDF2`
  
You can install the required libraries using:

```bash
pip install -r requirements.txt
```

## Configuration

Before running the application, make sure to configure the following parameters in your configuration file or command line arguments:

- `pdf_path`: Path to the PDF document to process.
- `questions`: Comma-separated list of questions to ask.
- `api_key`: Your OpenAI API key.
- `slack_token`: Slack API token for sending messages.
- `slack_channel`: Slack channel ID to post the messages.
- `model`:_optional_: Model to use for generating responses (default=gpt-4o-mini).
- `embed`:_optional_: Whether to use embeddings for pdf chunks for faster and cost-efficient retrieval using cosine-similarity (default=true).
- `embed_model`:_optional_: Embedding model to use (default=text-embedding-3-small).
- `chunk_size`:_optional_: Size of each chunk when splitting the PDF (default=500).
- `chunk_overlap`:_optional_: Number of overlapping characters between chunks (default=100).
- `confidence_threshold`:_optional_: Confidence threshold for openapi responses (default=-1.5, can be fine-tuned).

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/PDF-QA.git
   cd PDF-QA
   ```

2. Run the script with the desired parameters:

   ```bash
   python main.py --questions "Comma-separated list of questions here" --pdf_path "path/to/pdf"
   ```

## Logging

Logs are recorded both in the console and in a log file. Ensure that the logging level is set according to your needs for debugging or monitoring in main.py.
