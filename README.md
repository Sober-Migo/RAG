---
title: RAG PDF Q&A
emoji: 📄
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
app_file: main.py
pinned: false
license: mit
short_description: Ask questions about any PDF — RAG with Chroma and Groq
---

# RAG — PDF Question Answering Assistant

Retrieval-Augmented Generation (RAG) app: upload a PDF, ask a question, and get an answer grounded only in that document.

![Demo](rag%20demo.gif)

## How it works

1. **Load & chunk** — `PyPDFLoader` reads the PDF; a recursive splitter cuts it into 1000-character chunks (100 overlap).
2. **Embed** — local all-MiniLM-L6-v2 (ONNX) via Chroma. No torch, no embedding API.
3. **Retrieve** — in-memory Chroma store returns the top-4 most similar chunks.
4. **Generate** — Groq (`llama-3.1-8b-instant` by default) answers using only the retrieved context, or says it doesn't know.
5. **Fallback** — with no `GROQ_API_KEY` (or if Groq fails), the app shows the most relevant passages instead (extractive mode).

## Stack

Python **3.11** · Gradio 4.44 · LangChain 0.2 · ChromaDB 0.4 · pypdf · Groq

## Project structure

```
main.py              # entry point
app/
  config.py          # env-based settings
  documents.py       # PDF loading + chunking
  embeddings.py      # local ONNX embeddings
  vectorstore.py     # Chroma store + retriever
  llm.py             # Groq LLM wrapper
  chain.py           # RAG chain + extractive fallback
  ui.py              # Gradio interface
```

## Run locally

Requires **Python 3.11**.

```bash
git clone https://github.com/Sober-Migo/RAG.git
cd RAG
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your GROQ_API_KEY
python main.py            # opens on http://localhost:7860
```

Get a free Groq key at <https://console.groq.com/keys>.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required for generated answers |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `CHUNK_SIZE` | `1000` | Chunk size in characters |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TEMPERATURE` | `0.3` | Sampling temperature |
| `MAX_NEW_TOKENS` | `256` | Max answer length |
| `EXTRACTIVE_FALLBACK` | `true` | Show passages if Groq is unavailable |
| `SERVER_NAME` / `SERVER_PORT` | `0.0.0.0` / `7860` | Gradio server settings |

## Deploy on Hugging Face Spaces

The YAML header at the top of this file configures the Space (Gradio SDK, Python 3.11, entry point `main.py`).

1. Push this repo to a Space with SDK **Gradio**.
2. In **Settings → Variables and secrets**, add a secret named `GROQ_API_KEY`.
3. The Space builds and starts automatically.

## License

MIT
