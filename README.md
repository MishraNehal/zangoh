# Customer Support Ticket Agent

A guided starter project for building a text-based support agent using FastAPI, Streamlit, an open-source LLM, RAG, and a mock ticket tool.

Read the separate Assignment Implementation Guide before changing the starter.

## Architecture

```text
Streamlit -> FastAPI -> SupportPipeline -> LangGraph workflow
                                      |-> RAG/Chroma knowledge
                                      \-> session-bound ticket tool
```

## What Is Provided

- Request, response, and ticket models.
- A LangChain `ChatOpenAI` binding for an OpenAI-compatible open-source model.
- A typed LangGraph state, node skeleton, and routing graph.
- Document loading, splitting, embeddings, and Chroma component bindings.
- Session-state data structure.
- In-memory ticket repository with duplicate protection.
- Four support-policy documents.
- FastAPI and Streamlit scaffolding.
- RAG and agent integration TODOs.
- Initial repository and session tests.

## Candidate Work

Complete the TODOs in:

- `src/rag/retriever.py`
- `src/llm/workflow.py`
- `src/pipeline.py`
- `streamlit_app.py`

You may add or reorganize files when the resulting design remains clear and testable.

## Requirements

- Python 3.11 or newer.
- An OpenAI-compatible endpoint serving an open-source instruction model, or an equivalent open-source model integration.
- Enough local space for the selected embedding model and vector index.

## Setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

### macOS

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Configure `.env` for the selected model endpoint. Do not commit credentials.

## Run

Start the API:

```sh
uvicorn src.api.server:app --reload --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}"
```

In another terminal, start the UI:

```sh
streamlit run streamlit_app.py --server.address "${STREAMLIT_HOST:-127.0.0.1}" --server.port "${STREAMLIT_PORT:-8501}"
```

The default UI is `http://localhost:8501`; FastAPI documentation is `http://localhost:8000/docs`. Override both ports through `.env` and the corresponding command-line values when necessary.

## Test

```sh
pytest -q
```

Useful manual requests:

```sh
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-1","message":"How long does standard shipping take?"}'
```

For Windows PowerShell, use `Invoke-RestMethod` or place the equivalent JSON request in FastAPI's `/docs` interface.

## Completion Checklist

- RAG answers use the supplied documents and return source names.
- Unknown answers are not fabricated.
- Ticket details are collected over multiple turns.
- Tickets are created only after validation.
- Repeated requests in one session do not create duplicate tickets.
- API and Streamlit failures are displayed clearly.
- Automated tests cover the principal success and failure paths.
- This README is updated with the candidate's final architecture, provider choices, platform notes, and troubleshooting guidance.
