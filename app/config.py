"""Configuration — free-friendly defaults for Kaggle."""
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# LLM via HF router (may require Inference Providers access / small free quota)
LLM_MODEL = os.getenv("LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta")

# If LLM call fails, return retrieved chunks only (always free)
EXTRACTIVE_FALLBACK = os.getenv("EXTRACTIVE_FALLBACK", "true").lower() in (
    "1",
    "true",
    "yes",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

SERVER_NAME = os.getenv("SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
