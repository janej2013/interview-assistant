import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model Settings
    EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-ada-002"
    LLM_MODEL = "gpt-3.5-turbo"  # or "gpt-4"
    LLM_TEMPERATURE = 0.0

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
