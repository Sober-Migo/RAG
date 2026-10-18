"""Central configuration. Override via environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

# Embeddings (local sentence-transformers — free, no API key)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# LLM via Hugging Face
# - Default: small local-friendly instruct model
# - For better quality set HF_TOKEN and use a larger model, e.g.:
#   HuggingFaceH4/zephyr-7b-beta  or  mistralai/Mistral-7B-Instruct-v0.2
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")

# Use HF Inference API when True and HF_TOKEN is set; otherwise local pipeline
USE_INFERENCE_API = os.getenv("USE_INFERENCE_API", "false").lower() in ("1", "true", "yes")

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Generation
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

# Gradio
SERVER_NAME = os.getenv("SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
