"""Global settings for the project."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Processing
MAX_WORKERS = 1

# NLP
CROSSENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
OPENAI_LARGE_MODEL = "gpt-3.5-turbo-16k"
OPENAI_QUALITY_MODEL = "gpt-4"
OPENAI_FAST_MODEL = "gpt-3.5-turbo"

CLUSTER_DESCRIPTION_MODEL = "gpt-3.5-turbo"
OPENAI_REQUEST_TIMEOUT = 500
API_RETRY_ATTEMPTS = 5
RANDOM_SEED = 42
MAX_SAMPLES = 800


# Environment variables. Set these at the environment level to revealing secure details.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set.")
