# RAG — PDF Question Answering

Upload a PDF → retrieve relevant passages → (optional) generate an answer.

## Important (2026)

`api-inference.huggingface.co` **no longer exists** (DNS fails / 410 Gone).  
This repo does **not** call that host.

| Component | How (free) |
|-----------|------------|
| **Embeddings** | Chroma default ONNX `all-MiniLM-L6-v2` — **fully free, local, no HF API** |
| **Retrieval** | Chroma similarity — free |
| **Generation** | Optional via `huggingface_hub.InferenceClient` (router). Free quota is limited; if it fails, app falls back to **extractive mode** (returns top passages) |

---

## Kaggle (recommended)

```python
import shutil, os
if os.path.exists("/kaggle/working/RAG"):
    shutil.rmtree("/kaggle/working/RAG")
!git clone https://github.com/Sober-Migo/RAG.git /kaggle/working/RAG
%cd /kaggle/working/RAG

!pip install -q gradio==4.44.0 langchain==0.2.11 langchain-core==0.2.43 \
  langchain-community==0.2.10 langchain-text-splitters==0.2.4 \
  chromadb==0.4.24 pypdf==4.3.1 pydantic==2.9.1 \
  "huggingface_hub>=0.26.0" python-dotenv==1.0.1 onnxruntime tokenizers

# Restart session once after install, then:
import os, sys
from getpass import getpass
# Optional — only needed for generative answers
os.environ["HF_TOKEN"] = getpass("HF token (Enter to skip → extractive only): ") or ""

sys.path.insert(0, "/kaggle/working/RAG")
from app.ui import create_app
create_app().launch(share=True, server_name="0.0.0.0")
```

Without a token (or if the router rejects free accounts), the app still works: it returns the **most relevant PDF passages** for your question.

---

## Architecture

```text
PDF → split → ONNX embeddings → Chroma → top-k chunks
                 ↘ optional HF router LLM → answer
                 ↘ else extractive passages (always free)
```

## License

MIT
