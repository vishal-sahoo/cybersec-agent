# Cybersecurity Agent

A local, retrieval‑augmented (RAG) question‑answering agent for cybersecurity knowledge. It indexes your own `.txt` notes into a vector database and answers questions about them from the command line, grounding every response in your documents rather than the model's memory alone.

Built with [LlamaIndex](https://www.llamaindex.ai/), [ChromaDB](https://www.trychroma.com/), Google Gemini, and local HuggingFace embeddings.

## How it works

1. **Load** — reads all `.txt` files from the [`data/`](data/) directory.
2. **Embed** — turns them into vectors with the local `BAAI/bge-small-en-v1.5` HuggingFace model (runs on CPU, no API needed for embeddings).
3. **Store** — persists the vectors in a ChromaDB store (`./chroma_db_store`).
4. **Query** — retrieves the most relevant passages and asks Google Gemini to answer, using only that retrieved context.

The included sample document (`data/enumeration_basics.txt`) covers website enumeration basics, so out of the box you can ask the agent things like *"What tools are used for enumeration?"* Drop in your own notes to expand its knowledge.

## Requirements

- Python 3.10+
- A [Google AI Studio API key](https://aistudio.google.com/app/apikey) for Gemini

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API key:

```
GOOGLE_API_KEY="your_key_here"
```

## Usage

```bash
python simple_agent.py
```

The agent loads and indexes your documents, then drops into an interactive prompt:

```
Your query: What tools are commonly used for website enumeration?
```

Type `exit` to quit.

## Adding your own knowledge

Place any `.txt` files into the [`data/`](data/) directory and restart the agent — they'll be indexed automatically on the next run.

## Tech stack

| Component | Choice |
| --- | --- |
| Orchestration | LlamaIndex |
| LLM | Google Gemini (`gemini-pro`) |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace, local/CPU) |
| Vector store | ChromaDB (persistent) |
| Config | `python-dotenv` |
