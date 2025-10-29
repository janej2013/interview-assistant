import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Model Settings
    EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-ada-002"
    LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    OPEN_SOURCE_MODEL = os.getenv(
        "OPEN_SOURCE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"
    )
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "open_source").lower()
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
    LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))

    # Chunking Strategy
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Retrieval Settings
    TOP_K = 4  # Number of documents to retrieve
    SIMILARITY_THRESHOLD = 0.7

    # Vector Store
    VECTOR_STORE_PATH = "./data/vectorstore"
    COLLECTION_NAME = "interview_stories"

    # Evaluation
    EVAL_DATASET_PATH = "./data/eval_questions.json"


config = Config()
