# RAG — PDF Question Answering

Retrieval-Augmented Generation (RAG) app: upload a PDF, ask questions, get answers grounded in the document.

Built from the IBM *Generative AI Engineering with LLMs* specialization (RAG + LangChain track), reimplemented with a **pure Hugging Face** stack (no IBM Watsonx / watsonx.ai API).

---

## Architecture

```text
PDF upload
    → PyPDFLoader
    → RecursiveCharacterTextSplitter (chunk_size / overlap)
    → Hugging Face embeddings (sentence-transformers)
    → Chroma vector store
    → Retriever (top-k similarity)
    → RetrievalQA (stuff chain)
    → Hugging Face LLM (local pipeline or Inference API)
    → Gradio UI
```

| Stage | Implementation |
|--------|----------------|
| Loader | `langchain_community.document_loaders.PyPDFLoader` |
| Splitter | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector DB | Chroma (in-memory per request) |
| LLM | `google/flan-t5-base` local pipeline by default |
| UI | Gradio |

---

## Repository layout

```text
RAG/
├── main.py                 # python main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── app/
    ├── __init__.py
    ├── config.py           # models, chunk size, ports (env overrides)
    ├── embeddings.py       # HF sentence-transformers
    ├── llm.py              # HF pipeline or Inference API
    ├── documents.py        # PDF load + split
    ├── vectorstore.py      # Chroma + retriever
    ├── chain.py            # RetrievalQA
    └── ui.py               # Gradio interface
```

---

## Quick start

```bash
git clone https://github.com/Sober-Migo/RAG.git
cd RAG

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

Open the URL Gradio prints (default `http://127.0.0.1:7860`).

### Optional: Hugging Face token

For gated models or the Inference API:

```bash
cp .env.example .env
# edit .env → set HF_TOKEN=hf_...
```

```bash
# Example: use Inference API + a stronger instruct model
export USE_INFERENCE_API=true
export HF_TOKEN=hf_xxx
export LLM_MODEL=HuggingFaceH4/zephyr-7b-beta
python main.py
```

---

## Configuration

| Variable | Default | Meaning |
|----------|---------|----------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `LLM_MODEL` | `google/flan-t5-base` | Local or API model id |
| `USE_INFERENCE_API` | `false` | `true` → HF Inference API |
| `HF_TOKEN` | — | Token for API / gated models |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TEMPERATURE` | `0.3` | Generation temperature |
| `MAX_NEW_TOKENS` | `256` | Max generated tokens |
| `SERVER_PORT` | `7860` | Gradio port |

---

## Design notes

- **No Watsonx** — original course labs used `ibm-watsonx-ai` / `langchain-ibm`. This repo uses only open Hugging Face models so it runs without an IBM Cloud project.
- **Query is respected** — the course sample hardcoded `"Summarize this document"`; this app passes the user’s question through to the chain.
- **Modular layout** — each stage (load, split, embed, retrieve, generate, UI) is a separate module for clarity and testing.
- **Demo** — a hosted demo (Hugging Face Space / similar) can be added later without changing the core pipeline.

---

## Origin

Course path: **IBM Generative AI Engineering with LLMs** (Coursera)  
→ C06 *Fundamentals of AI Agents Using RAG and LangChain*  
→ C07 *Project: Generative AI Applications with RAG and LangChain*

Companion materials (full specialization labs):  
[IBM_Generative_AI_Engineering_with_LLMs_Specialization](https://github.com/Sober-Migo/IBM_Generative_AI_Engineering_with_LLMs_Specialization)

---

## License

MIT
