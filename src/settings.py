"""Global settings for the project."""

from __future__ import annotations

import os
from dotenv import load_dotenv
from loguru import logger


load_dotenv()

# Processing
MAX_WORKERS = 1

# NLP
CROSSENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PALM_EMBEDDING_MODEL = "models/embedding-gecko-001"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_LARGE_MODEL = "gpt-3.5-turbo-16k"
OPENAI_QUALITY_MODEL = "gpt-4"
OPENAI_FAST_MODEL = "gpt-3.5-turbo"

PALM_MODEL = "models/text-bison-001"
CLUSTER_DESCRIPTION_MODEL = "gpt-3.5-turbo"
OPENAI_REQUEST_TIMEOUT = 500
API_RETRY_ATTEMPTS = 5
RANDOM_SEED = 42
MAX_SAMPLES = 1000

# Search Console Drive
SERVICE_ACCOUNT_CREDENTIALS = "service.credentials.json"
SERVICE_ACCOUNT_SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]
SERVICE_ACCOUNT_SUBJECT = "clients@locomotive.agency"


# Environment variables. Set these at the environment level to revealing secure details.
try:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    PALM_API_KEY = os.environ["PALM_API_KEY"]
except KeyError as e:
    logger.error("Environment variable not set: {}", str(e))
