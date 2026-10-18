"""Central configuration. HF Inference API only (no local models)."""
import os
from dotenv import load_dotenv

load_dotenv()

# Required for Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Models served via Hugging Face Inference API
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "HuggingFaceH4/zephyr-7b-beta",
)

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Generation
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

# Gradio
SERVER_NAME = os.getenv("SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
