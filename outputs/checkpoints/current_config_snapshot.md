# Current Config Snapshot

Checkpoint date: 2026-06-28

## Scope

This checkpoint records the currently runnable Streamlit configuration. No business logic, paper content, chunks, or vector index data was changed. `data/raw_papers/` was not read.

## Application Startup

Direct startup:

```powershell
streamlit run src/app/streamlit_app.py
```

Project wrapper startup:

```powershell
python scripts/run_app.py
```

The wrapper defaults to `127.0.0.1:8501`, disables Streamlit file watching, and disables browser usage statistics. The current application URL is `http://127.0.0.1:8501/` (equivalent to `http://localhost:8501/` locally).

## Active Knowledge Store Configuration

- Chunks file: `data/chunks/chunks.jsonl`
- Current chunks count: 1,864
- Configured Chroma path: `vector_store/chroma`
- Chroma path type: junction
- Junction runtime target: `C:\Users\JZ\.cache\injection_molding_rag_agent\chroma`
- Collection name: `injection_molding_chunks`
- Current collection count: 1,864
- Embedding dimension: 1,024

The Streamlit app obtains these vector-store settings from `configs/retrieval_config.yaml`. BM25 reads the same canonical chunks file, and Dense retrieval opens the named Chroma collection.

## Model Configuration

- LLM provider: `ollama`
- LLM model: `qwen2.5:7b`
- Ollama base URL: `http://localhost:11434`
- Embedding provider: `sentence-transformers`
- Embedding model: `E:/AI_Models/BAAI/bge-m3`
- Embedding local path: `E:/AI_Models/BAAI/bge-m3`
- Paid OpenAI API: not enabled

## Smoke Check

- Core app, Agent, and retrieval modules import successfully.
- Chunks file exists and is non-empty.
- Configured Chroma path exists.
- Collection `injection_molding_chunks` exists and has more than zero records.
- Local embedding model directory exists.
- Ollama configuration exists and the local `/api/tags` endpoint responded successfully.
- Streamlit endpoint responded successfully on port 8501.

## Pending Audit

This checkpoint confirms that the current knowledge store is runnable, but it does not establish how many source papers contributed to the active chunks and Chroma collection. The next task is to audit whether the active knowledge base was built only from the 30-paper development set.

## Version Control Status

The project directory is not currently a Git repository. Initialize it before creating a source-control checkpoint:

```powershell
git init
```
