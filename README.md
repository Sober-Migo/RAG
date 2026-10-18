# RAG — PDF Question Answering

Upload a PDF, ask questions grounded in the document.

## Stack (2026)

| Stage | Implementation |
|--------|----------------|
| Embeddings | Chroma ONNX `all-MiniLM-L6-v2` (local, free, no torch) |
| Vector DB | Chroma |
| LLM | `huggingface_hub.InferenceClient` + small instruct model |
| UI | Gradio |

**Do not use** `api-inference.huggingface.co` — that host is gone.  
Requires **`huggingface_hub>=0.28`** so the client talks to the current router.

Default model: `Qwen/Qwen2.5-1.5B-Instruct` (often works with a free Read token).  
If generation fails, the app falls back to **extractive retrieval** (always free).

---

## Kaggle

```python
import shutil, os
if os.path.exists("/kaggle/working/RAG"):
    shutil.rmtree("/kaggle/working/RAG")
!git clone https://github.com/Sober-Migo/RAG.git /kaggle/working/RAG
%cd /kaggle/working/RAG

!pip install -q -U "huggingface_hub>=0.28.0"
!pip install -q gradio==4.44.0 langchain==0.2.11 langchain-core==0.2.43 \
  langchain-community==0.2.10 langchain-text-splitters==0.2.4 \
  chromadb==0.4.24 pypdf==4.3.1 pydantic==2.9.1 \
  python-dotenv==1.0.1 onnxruntime tokenizers

# Restart session once, then:
import os, sys
from getpass import getpass
os.environ["HF_TOKEN"] = getpass("HF token: ")

sys.path.insert(0, "/kaggle/working/RAG")
from app.ui import create_app
create_app().launch(share=True, server_name="0.0.0.0")
```

Optional model override:

```python
os.environ["LLM_MODEL"] = "Qwen/Qwen2.5-1.5B-Instruct"
```

---

## Origin

IBM Generative AI Engineering with LLMs (Coursera) RAG project, without Watsonx.

## License

MIT
