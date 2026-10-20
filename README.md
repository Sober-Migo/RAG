---
title: RAG PDF Q&A
emoji: 📄
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.11"
app_file: main.py
pinned: false
license: mit
short_description: Ask questions about any PDF — RAG with Chroma and Groq
---

# RAG — PDF Question Answering Assistant

Retrieval-Augmented Generation (RAG) app: upload a PDF, ask a question, and get an answer grounded only in that document.

**Live Space:** [huggingface.co/spaces/Sober-Migo/RAG](https://huggingface.co/spaces/Sober-Migo/RAG)

![Demo](rag%20demo.gif)

## How it works

1. **Load & chunk** — `PyPDFLoader` reads the PDF; a recursive splitter cuts it into 1000-character chunks (100 overlap).
2. **Embed** — local all-MiniLM-L6-v2 (ONNX) via Chroma. No torch, no embedding API.
3. **Retrieve** — in-memory Chroma store returns the top-k most similar chunks.
4. **Generate** — Groq (`llama-3.1-8b-instant` by default) answers using only the retrieved context.
5. **Fallback** — with no `GROQ_API_KEY` (or if Groq fails), the app shows the most relevant passages (extractive mode).

## Stack

Python **3.11** · Gradio 4.44 · LangChain 0.2 · ChromaDB 0.4 · pypdf · Groq · ONNX Runtime

## Project structure

```
main.py              # HF Spaces + local entry (exports `demo`)
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

```bash
git clone https://github.com/Sober-Migo/RAG.git
cd RAG
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add GROQ_API_KEY=
python main.py            # http://localhost:7860
```

Free Groq key: <https://console.groq.com/keys>

## Hugging Face Space checklist

1. Space URL: <https://huggingface.co/spaces/Sober-Migo/RAG>
2. **Settings → Variables and secrets** → add secret **`GROQ_API_KEY`** (required for generated answers).
3. **Settings → Factory rebuild** (or unpause / restart) after setting the secret.
4. Wait for the build log to finish; open the Space and upload a short PDF.

GitHub Actions workflow `.github/workflows/sync_to_hf.yml` can mirror `main` → the Space when `HF_TOKEN` is set as a GitHub secret.

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

## License

MIT
