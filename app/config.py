"""Configuration — Groq for generation, local ONNX embeddings."""
import os
from dotenv import load_dotenv

load_dotenv()

# Required for generative answers
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# If no Groq key / call fails → return retrieved passages
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
