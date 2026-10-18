"""Configuration."""
import os
from dotenv import load_dotenv

load_dotenv()

# Preferred free generative path
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Hugging Face OpenAI-compatible router
# base_url=https://router.huggingface.co/v1  (NOT api-inference.huggingface.co)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_ROUTER_BASE_URL = os.getenv(
    "HF_ROUTER_BASE_URL", "https://router.huggingface.co/v1"
)
# Model that worked for you on the free router path
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

EXTRACTIVE_FALLBACK = os.getenv("EXTRACTIVE_FALLBACK", "true").lower() in (
    "1",
    "true",
    "yes",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

SERVER_NAME = os.getenv("SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
