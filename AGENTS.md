# Repository Guidelines

## Project Structure & Module Organization
- Core logic lives in `src/`. Key modules: `document_loader.py` (ingestion), `chunking.py` (splitting), `vectorstore.py` (Chroma management), `qa_chain.py` and `retriever.py` (LLM + retrieval), and `interview_assistant.py` for practice flows.
- `main.py` bootstraps the full RAG pipeline over stories in `data/raw/`; `interview_main.py` loads prepared answers from `data/prepared_answers.txt` or `data/prepared_answers/`.
- Configuration constants and environment access are in `config/config_setup.py`. Create `.env` alongside the repo root with `OPENAI_API_KEY=...`.
- Generated artifacts (vector stores, evaluation data) are expected under `data/`; keep large blobs out of version control.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create and activate a local virtual environment.
- `pip install -r updated_requirements.txt` — install the LangChain, Chroma, and evaluation dependencies pinned for this project.
- `python main.py` — run the RAG QA setup, rebuild chunks/vector store if `data/raw/` changes.
- `python interview_main.py` — launch the interview assistant workflow that scores prepared answers and runs interactive practice.

## Coding Style & Naming Conventions
- Use 4-space indentation, follow PEP 8, and mirror existing docstring style for public methods and modules.
- Keep type hints on function signatures (existing modules annotate return types) and prefer descriptive class names over abbreviations.
- Constants stay uppercase within `config/`; runtime paths should be passed as arguments rather than hardcoded duplicates.
- When introducing new chains or loaders, place them in `src/` with filenames matching snake_case module names.

## Testing Guidelines
- Add new tests under `tests/` (create the folder if needed) and exercise high-level flows via `pytest`. Run with `pytest -q` before opening a PR.
- Use `DocumentLoader.load_from_string` for lightweight fixtures instead of relying on large files; drop temporary data under `data/tmp/` and git-ignore it.
- Validate retrieval accuracy with sample documents in `data/raw/` and summarize precision/recall observations in the PR description when altering chunking or retriever logic.

## Commit & Pull Request Guidelines
- Follow the short imperative style seen in history (e.g., “Add vectorstore utilities”, “Refine chunking stats”). Group related edits per commit.
- Include in each PR: purpose summary, testing evidence (`pytest` run and manual command outputs), and notes on data migrations (e.g., whether contributors must delete `data/vectorstore/`).
- Link to any issue or discussion that triggered the change, attach relevant console snippets instead of screenshots, and call out secrets management steps (confirm `.env` remains local).
