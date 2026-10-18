# RAG — PDF Question Answering

Upload a PDF → retrieve relevant passages → generate an answer.

## LLM priority (free-first)

1. **Groq** (`GROQ_API_KEY`) — free tier, recommended  
2. **Hugging Face** (`HF_TOKEN`) — often needs Inference Providers credits  
3. **Extractive** — no key; returns top retrieved passages  

Embeddings are always **local ONNX** (Chroma MiniLM) — free, no API.

---

## Kaggle

1. Create a free Groq key: https://console.groq.com/keys  
2. Run:

```python
import shutil, os
if os.path.exists("/kaggle/working/RAG"):
    shutil.rmtree("/kaggle/working/RAG")
!git clone https://github.com/Sober-Migo/RAG.git /kaggle/working/RAG
%cd /kaggle/working/RAG

!pip install -q -U "huggingface_hub>=0.28.0" groq
!pip install -q gradio==4.44.0 langchain==0.2.11 langchain-core==0.2.43 \
  langchain-community==0.2.10 langchain-text-splitters==0.2.4 \
  chromadb==0.4.24 pypdf==4.3.1 pydantic==2.9.1 \
  python-dotenv==1.0.1 onnxruntime tokenizers

# Restart session once after install
```

```python
import os, sys
from getpass import getpass

os.environ["GROQ_API_KEY"] = getpass("Groq API key: ")
# optional: os.environ["GROQ_MODEL"] = "llama-3.1-8b-instant"

sys.path.insert(0, "/kaggle/working/RAG")
for m in list(sys.modules):
    if m == "app" or m.startswith("app."):
        del sys.modules[m]

from app.ui import create_app
create_app().launch(share=True, server_name="0.0.0.0")
```

---

## Architecture

```text
PDF → chunk → ONNX embeddings → Chroma → top-k context
                                    ↓
                         Groq (or HF) chat LLM
                                    ↓
                              Gradio answer
```

## Origin

IBM Generative AI Engineering with LLMs (Coursera) RAG project, without Watsonx.

## License

MIT
