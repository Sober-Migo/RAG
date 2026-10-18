# RAG — PDF Question Answering

Upload a PDF, ask questions, get answers grounded in the document.

**Stack:** Hugging Face **Inference API** (embeddings + LLM) · Chroma · LangChain · Gradio  
**No local GPU / torch / transformers** — runs cleanly on Kaggle & Colab.

Based on the IBM *Generative AI Engineering with LLMs* specialization (RAG + LangChain), with Watsonx replaced by Hugging Face.

---

## Architecture

```text
PDF → PyPDFLoader → chunk → HF Inference embeddings → Chroma
  → retriever → prompt → HF Inference LLM → Gradio
```

| Stage | Implementation |
|--------|----------------|
| Embeddings | HF Inference API (`sentence-transformers/all-MiniLM-L6-v2`) |
| LLM | HF Inference API (`HuggingFaceH4/zephyr-7b-beta` default) |
| Vector DB | Chroma (in-memory) |
| UI | Gradio 4.44 |

---

## Requirements

1. A free [Hugging Face token](https://huggingface.co/settings/tokens) (`read` is enough for public models).
2. Python 3.10+ (Kaggle/Colab 3.11 is fine).

---

## Kaggle / Colab (recommended)

```python
# 1) Clone
!git clone https://github.com/Sober-Migo/RAG.git
%cd RAG
!git pull

# 2) Install — no torch
!pip install -q -r requirements.txt

# 3) Token
import os
from getpass import getpass
os.environ["HF_TOKEN"] = getpass("HF token: ")

# 4) Launch
import sys
sys.path.insert(0, ".")
from app.ui import create_app
demo = create_app()
demo.launch(share=True, server_name="0.0.0.0")
```

---

## Local

```bash
git clone https://github.com/Sober-Migo/RAG.git
cd RAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add HF_TOKEN=
python main.py
```

---

## Config

| Variable | Default |
|----------|---------|
| `HF_TOKEN` | **required** |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `LLM_MODEL` | `HuggingFaceH4/zephyr-7b-beta` |
| `CHUNK_SIZE` | `1000` |
| `MAX_NEW_TOKENS` | `256` |

If a model is gated, accept its license on the model page while logged into Hugging Face.

---

## Origin

IBM Generative AI Engineering with LLMs (Coursera) — C06/C07 RAG + LangChain project, reimplemented without Watsonx.

Companion labs: [IBM_Generative_AI_Engineering_with_LLMs_Specialization](https://github.com/Sober-Migo/IBM_Generative_AI_Engineering_with_LLMs_Specialization)

## License

MIT
