# Public Release Checkpoint

- checked_at: 2026-06-29T21:43:10+08:00
- repository: `injection-molding-rag-agent`
- remote: `origin https://github.com/JamaicaBoy/injection-molding-rag-agent.git`
- branch: `main`
- HEAD: `8beb806`
- working_tree: dirty (`21` tracked modifications, `48` untracked entries)

## Streamlit

- verified command: `.\.venv\Scripts\python.exe -m streamlit run src/app/streamlit_app.py`
- helper command: `.\.venv\Scripts\python.exe scripts/run_app.py`
- smoke test: PASS (`/_stcore/health` returned `ok` on temporary port `8510`)
- local app: `http://127.0.0.1:8501/`
- README note: two legacy examples still reference `app/streamlit_app.py`; the current entry point is `src/app/streamlit_app.py`.

## Corpus And Index

- corpus_mode: `full`
- effective_mode: `full`
- chunks_path: `data/chunks/full_chunks.jsonl`
- vector_persist_dir: `vector_store/chroma_full`
- collection_name: `injection_papers_full`
- fallback_mode: none

## Ignore Audit

The following release-sensitive paths were verified with `git check-ignore` and are ignored:

- `data/raw_papers/`
- `data/interim/`
- `data/processed/`
- `data/chunks/`
- `vector_store/`
- `artifacts/releases/`
- `artifacts/full_release_no_pdf_v1/`
- `.env`
- `.venv/`

No PDF content under `data/raw_papers/` was read during this audit. No files were staged, committed, or pushed.

## Release Readiness

The repository remote and runtime configuration are valid, but the working tree is not clean. Review the current tracked changes and untracked source/config/documentation files before creating the release checkpoint commit. Keep ignored corpora, vector stores, virtual environments, secrets, and generated release archives out of the normal Git commit.
