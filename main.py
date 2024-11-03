from config import *
import argparse
from agents.mainAgent import AIAgent
import logging

def create_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the logging level
    logger.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set level for console handler

    # Create a file handler
    file_handler = logging.FileHandler('logs/app.log')
    file_handler.setLevel(logging.INFO)  # Set level for file handler

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Set the formatter for both handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process parameters for the PDF answer extraction and Slack posting tool.")

    parser.add_argument("--questions", type=str,default=QUESTIONS, help="Comma-separated list of questions")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Size of each chunk when splitting the PDF.")
    parser.add_argument("--api_key", type=str, default=API_KEY, help="API key for accessing OpenAI or other services.")
    parser.add_argument("--slack_token", type=str, default=SLACK_TOKEN, help="Slack API token for sending messages.")
    parser.add_argument("--slack_channel", type=str, default=SLACK_CHANNEL,
                        help="Slack channel ID to post the messages.")
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP,
                        help="Number of overlapping characters between chunks.")
    parser.add_argument("--embed", type=bool, default=EMBED, help="Flag to enable or disable embedding.")
    parser.add_argument("--model", type=str, default=MODEL, help="Model to use for generating responses.")
    parser.add_argument("--pdf_path", type=str, default=PDF_PATH, help="Path to the PDF document to process.")
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL, help="Embedding model to use.")
    parser.add_argument("--confidence_threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold for openapi responses.")

    return parser.parse_args()


if __name__ == "__main__":
    # Instantiate Logger
    logger = create_logger()

    # Get all inputs from cmd
    cfg = parse_args()

    # Instantiate the main AI agent
    agent = AIAgent(cfg, logger)

    # Make a list of all questions
    questions = [q.strip() for q in cfg.questions.split(",")]

    # Run the process
    agent.process_and_respond(questions)






