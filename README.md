# RAG — PDF Question Answering

Upload a PDF → retrieve relevant passages → answer with **Groq**.

| Stage | Implementation |
|--------|----------------|
| Embeddings | Local ONNX `all-MiniLM-L6-v2` (Chroma, free) |
| Vector DB | Chroma |
| LLM | **Groq** (`llama-3.1-8b-instant` default) |
| Fallback | Extractive top-k passages if no Groq key |
| UI | Gradio |

No Hugging Face Inference / router / Watsonx.

---

## Kaggle

1. Free key: https://console.groq.com/keys  

```python
import shutil, os
if os.path.exists("/kaggle/working/RAG"):
    shutil.rmtree("/kaggle/working/RAG")
!git clone https://github.com/Sober-Migo/RAG.git /kaggle/working/RAG
%cd /kaggle/working/RAG

!pip install -q groq gradio==4.44.0 langchain==0.2.11 langchain-core==0.2.43 \
  langchain-community==0.2.10 langchain-text-splitters==0.2.4 \
  chromadb==0.4.24 pypdf==4.3.1 pydantic==2.9.1 \
  python-dotenv==1.0.1 onnxruntime tokenizers

# Restart session once after install, then:
from getpass import getpass
import os, sys
os.environ["GROQ_API_KEY"] = getpass("Groq API key: ")

sys.path.insert(0, "/kaggle/working/RAG")
from app.ui import create_app
create_app().launch(share=True, server_name="0.0.0.0")
```

---

## Local

```bash
git clone https://github.com/Sober-Migo/RAG.git && cd RAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set GROQ_API_KEY=
python main.py
```

---

## Origin

IBM Generative AI Engineering with LLMs (Coursera) RAG project, reimplemented with Groq + local embeddings.

## License

MIT
