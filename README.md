# Customer Support Ticket Agent

A text-and-voice customer-support agent built on FastAPI, Streamlit, LangGraph,
and RAG over a small Markdown knowledge base, with a mock ticket-creation tool.

It answers policy questions grounded in the supplied knowledge base, collects
the fields needed to raise a support ticket over a multi-turn conversation,
creates the ticket through a mock in-memory ticket service, and lets a
reviewer either type or *speak* a message and either read or *hear* the
agent's reply.

## Architecture

```text
                        ┌─────────────────────────┐
   (typed)  ──────────▶ │                         │
                        │      Streamlit UI       │
   (spoken) ─┐          │                         │
             │          └────────────┬────────────┘
             │                       │ POST /chat  {session_id, message}
             ▼                       ▼
      ┌─────────────┐        ┌───────────────────────────────────────┐
      │  mic input  │        │                FastAPI                │
      └──────┬──────┘        │  /health   /chat   /tickets/{id}       │
             │               │  /voice/transcribe   /voice/synthesize │
   POST      │               └───────┬───────────────────┬───────────┘
  /voice/            process(session_id, message)         │
  transcribe         │                                    │ synthesize(text)
             │        ▼                                    ▼
             │  ┌───────────────────────┐         ┌─────────────────────┐
             │  │    SupportPipeline     │         │     VoicePipeline    │
             │  └───────────┬────────────┘         │  (STT adapter,       │
             │              │                       │   TTS adapter)       │
             │              ▼                       └──────────┬──────────┘
             │   ┌─────────────────────────┐                   │
             └──▶│   LangGraph workflow     │        Groq Whisper (STT)
   (edited        │  retrieve → decide →     │        Edge TTS      (TTS)
   transcript,     │  (answer | ticket)       │
   same /chat       └─────┬──────────┬───────┘
   request)                │          │
                           ▼          ▼
                 ┌──────────────┐  ┌──────────────────┐
                 │ RAG / Chroma │  │  Mock ticket tool  │
                 │ (knowledge_  │  │ (in-memory repo,   │
                 │  base/*.md)  │  │  session-idempotent)│
                 └──────────────┘  └──────────────────┘
```

Voice is a thin layer bolted onto the *edges* of this diagram, not a second
pipeline: STT only ever produces text that is handed to the exact same
`POST /chat` flow a typed message would use, and TTS only ever consumes the
exact text `/chat` already returned. `SupportPipeline` (RAG, the LangGraph
agent, session state, ticket creation) has no knowledge that voice exists.

## Required Technology

- **Language:** Python 3.11+
- **API:** FastAPI
- **UI:** Streamlit
- **LLM:** Groq-hosted `openai/gpt-oss-20b` (open-weight, hosted inference),
  via LangChain's `ChatOpenAI` against Groq's OpenAI-compatible endpoint
- **Vector DB:** ChromaDB (via `langchain-chroma`), persisted locally
- **Orchestration:** LangGraph (`retrieve → decide → answer | ticket`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, open-source, no API key)
- **Voice (mid-session addition):** Groq-hosted Whisper for STT, Edge TTS for TTS — see below

### Provider choices and rationale

**LLM — Groq `openai/gpt-oss-20b`.** Chosen for fast hosted inference on an
open-weight model. `gpt-oss` is a *reasoning* model, so `src/llm/client.py`
passes `extra_body={"reasoning_effort": "low"}` (a Groq-specific parameter,
not part of the OpenAI spec, hence `extra_body` rather than a first-class
`ChatOpenAI` kwarg) to keep chain-of-thought short for a support reply.

The `decide` node in `src/llm/workflow.py` binds `SupportDecision` (a Pydantic
model) via `model.with_structured_output(SupportDecision, method="json_schema")`.
LangChain's default structured-output method for OpenAI-compatible models is
function/tool-calling, which `gpt-oss` on Groq does not reliably honor for
this kind of routing decision — it would occasionally return the fields as
free text instead of a tool call. Forcing `method="json_schema"` makes Groq
constrain the raw completion to the schema at the API level, which removes
that failure mode entirely.

**STT — Groq-hosted Whisper (`whisper-large-v3-turbo`), called directly over
HTTP.** The project already talks to Groq (an OpenAI-compatible endpoint) for
the chat model, so this reuses the same account and API key — no second
provider signup. Rather than pulling in an SDK, `src/voice/stt_groq.py` makes
one `httpx` multipart `POST` to Groq's `/audio/transcriptions` endpoint, which
is enough for a single call and keeps the adapter to "core Python audio and
HTTP libraries" as required by the mid-session assignment.

**TTS — Edge TTS (`edge-tts` package, no API key).** Chosen so the voice
output half of the feature needs zero new secrets and works for any reviewer
without a paid TTS account, while still producing natural-sounding neural
speech. `src/voice/tts_edge.py` wraps `edge_tts.Communicate` and returns
`audio/mpeg` bytes.

Both STT and TTS are implemented behind the supplied abstract
`STTService`/`TTSService` contracts (`src/voice/contracts.py`) and constructed
through `src/voice/factory.py`, so either could be swapped for a different
provider later by editing only that one file.

## What's Implemented

- `src/rag/retriever.py` — Chroma-backed retrieval with stable, content-hash
  chunk IDs so repeated `initialize()` calls (e.g. every API restart) never
  re-embed or duplicate existing chunks; cosine similarity space set
  explicitly (`hnsw:space: cosine`); a relevance-score threshold (`0.2`) below
  which a chunk is treated as "not actually relevant" rather than forced into
  the prompt.
- `src/llm/workflow.py` — the LangGraph agent: `retrieve → decide → (answer |
  ticket)`. `decide()` uses structured output as described above. `answer()`
  grounds strictly in retrieved chunks and explicitly declines to fabricate
  when nothing relevant was retrieved. `collect_or_create()` asks one focused
  follow-up question per missing ticket field, validates through Pydantic
  before ever calling the ticket tool, and is idempotent per session (retrying
  a completed request returns the existing ticket rather than creating a new
  one).
- `src/pipeline.py` — binds session state to the compiled workflow and
  translates internal failures to `ComponentNotReadyError` (503) /
  `AgentProcessingError` (502) at the API boundary.
- `src/tools/ticket_tool.py` — in-memory `TicketRepository` with
  session-to-ticket idempotency, wrapped as a validated LangChain
  `StructuredTool` so the model can never supply the ticket ID, status, or
  timestamp itself.
- `src/api/server.py` — `GET /health`, `POST /chat`, `GET /tickets/{ticket_id}`,
  plus the mid-session `POST /voice/transcribe` and `POST /voice/synthesize`.
- `streamlit_app.py` — session-aware chat, source display, ticket
  confirmation, friendly error states for timeouts/422/503/malformed JSON,
  a microphone recorder with an editable transcript before it's sent, and a
  speaker icon on every agent response.
- `src/voice/` — the mid-session voice enhancement (STT/TTS contracts,
  orchestration pipeline, Groq Whisper + Edge TTS adapters). See
  **Mid-Session Change** below.
- Full test suite (`tests/`) — see **Tests**.

## Environment Variables

Copy `.env.example` to `.env` and fill in the values that need it. Nothing in
`.env.example` is a real credential.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible chat endpoint. Set to `https://api.groq.com/openai/v1` for Groq. |
| `LLM_API_KEY` | `not-required` | API key for the chat endpoint (Groq API key). |
| `LLM_MODEL` | `your-open-source-model` | Chat model identifier, e.g. `openai/gpt-oss-20b`. |
| `API_BASE_URL` | `http://localhost:8000` | Base URL the Streamlit app uses to reach FastAPI. |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | FastAPI bind address. |
| `STREAMLIT_HOST` / `STREAMLIT_PORT` | `127.0.0.1` / `8501` | Streamlit bind address. |
| `VECTOR_DB_PATH` | `.data/vector_db` | Local Chroma persistence directory. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model (downloaded from Hugging Face on first run). |
| `RAG_COLLECTION` | `customer-support` | Chroma collection name. |
| `RAG_TOP_K` | `3` | Number of chunks retrieved per query. |
| `STT_MODEL` | `whisper-large-v3-turbo` | Groq-hosted Whisper model used for transcription. |
| `STT_API_KEY` | *(empty)* | STT API key. Leave blank to reuse `LLM_API_KEY` (same Groq account); set only to use a separate key. |
| `TTS_VOICE` | `en-US-AriaNeural` | Edge TTS voice name. No API key needed for TTS. |

## Setup

Requires Python 3.11+, an internet connection (Groq for chat/STT, Hugging
Face Hub for the embedding model on first run, Microsoft's Edge TTS service
for speech synthesis), and a Groq API key.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env   # set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
```

### Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
${EDITOR:-nano} .env   # set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
```

### macOS

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
${EDITOR:-nano} .env   # set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
```

Minimum `.env` for Groq:

```dotenv
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...your_key...
LLM_MODEL=openai/gpt-oss-20b
```

`STT_API_KEY` can stay blank — it falls back to `LLM_API_KEY`. `TTS_VOICE`
can stay at its default. Never commit a filled-in `.env`.

## Run

Start the API (leave running in one terminal):

```sh
uvicorn src.api.server:app --reload --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}"
```

The first request that reaches `KnowledgeRetriever.initialize()` downloads
the embedding model from Hugging Face and builds the local Chroma index —
this happens once and is cached under `VECTOR_DB_PATH` after that.

In a second terminal, start the UI:

```sh
streamlit run streamlit_app.py --server.address "${STREAMLIT_HOST:-127.0.0.1}" --server.port "${STREAMLIT_PORT:-8501}"
```

Open `http://localhost:8501`. FastAPI's interactive docs are at
`http://localhost:8000/docs`.

## Using Voice

1. Open the **"🎤 Or record your message"** expander above the chat box.
2. Record a message with the browser microphone control.
3. Wait for **"Transcribing..."**; the recognized text appears in an editable
   box — correct it if needed.
4. Click **Submit transcript**. This sends the (possibly edited) text through
   the exact same `/chat` request a typed message uses — RAG, ticket
   collection, and session state all behave identically either way.
5. Once the agent replies, click **🔊 Play response** under that message to
   hear it. The button is disabled while audio is generating, and the audio
   is cached per message for the rest of the session (click **🔁 Replay**
   instead of re-generating it).

If transcription or synthesis fails, the app shows an inline error and the
typed-chat flow and the text response stay fully usable — voice failures
never remove or block the underlying text conversation.

## Test

```sh
pytest -q
```

Runs offline against fakes for the RAG store, the workflow, and both voice
adapters — no network or API key required. Three tests
(`tests/test_pipeline.py::test_live_*`) are gated behind
`@pytest.mark.skipif` on a real `LLM_API_KEY` and only run when one is
present in `.env`, since they make live Groq calls end-to-end.

Manual smoke checks once both servers are running:

```sh
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-1","message":"How long does standard shipping take?"}'

curl -X POST http://localhost:8000/voice/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"message_id":"demo-msg-1","text":"Your ticket has been created."}' \
  --output reply.mp3
```

(`/voice/transcribe` needs a multipart audio file, easiest to exercise from
the Streamlit UI or FastAPI's `/docs` page rather than a one-line `curl`.)

For Windows PowerShell, use `Invoke-RestMethod` or FastAPI's `/docs` page.

### What the tests cover

| Assignment requirement | Covered by |
|---|---|
| Policy question answered from RAG, with sources | `tests/test_rag.py`, live: `test_live_policy_question_is_grounded_and_cites_source` |
| Unknown question not fabricated | live: `test_live_unknown_question_does_not_fabricate` |
| Multi-turn ticket creation | live: `test_live_multi_turn_ticket_creation_and_retrieval` |
| Ticket retrieval | `tests/test_tickets.py` |
| Missing-field validation | `tests/test_session.py`, `tests/test_tickets.py` |
| Duplicate ticket protection | `tests/test_tickets.py::test_repository_prevents_duplicate_ticket_per_session` |
| Graceful model/backend failure | `tests/test_pipeline.py` (uninitialized pipeline, blank input) |
| Successful transcription (fake adapter) | `tests/test_voice_pipeline.py`, `tests/test_voice_api.py::test_transcribe_success` |
| Empty-audio validation | `tests/test_voice_pipeline.py::test_transcribe_rejects_empty_audio`, `test_voice_api.py::test_transcribe_rejects_empty_audio` |
| STT failure handling | `test_voice_api.py::test_transcribe_handles_stt_failure` |
| Successful synthesis (fake adapter) | `test_voice_api.py::test_synthesize_success` |
| TTS failure handling | `test_voice_api.py::test_synthesize_handles_tts_failure` |
| Voice-pipeline timing and media type | `test_voice_pipeline.py::test_voice_pipeline_contracts` |
| Typed-chat regression after voice integration | `test_voice_api.py::test_typed_chat_contract_is_unmodified_by_voice_routes` |
| Response ↔ playback audio association | `test_voice_api.py::test_voice_and_message_id_round_trip` |

## Mid-Session Change: Voice Input and Response Playback

Partway through the assignment, a mid-session requirement extended the
text-only agent with **microphone input** and **agent-response playback**,
without replacing or duplicating the text-agent workflow, and without using
any end-to-end conversational platform (Pipecat, LiveKit, Daily, Agora,
Twilio Voice, AssemblyAI Universal, Deepgram Aura, the OpenAI Realtime API,
or similar).

What changed, concretely:

- **New, isolated module:** `src/voice/` (contracts, orchestration, Groq
  Whisper adapter, Edge TTS adapter, factory) — nothing in `src/llm/`,
  `src/rag/`, `src/pipeline.py`, `src/tools/`, or `src/sessions/` was touched.
- **Two additive API routes:** `POST /voice/transcribe` and
  `POST /voice/synthesize` in `src/api/server.py`. `GET /health`,
  `POST /chat`, and `GET /tickets/{ticket_id}` keep their exact original
  request/response contracts.
- **Streamlit:** a microphone recorder feeding an editable transcript that is
  only sent through the *existing* `/chat` call after explicit confirmation,
  plus a speaker icon per agent response that calls `/voice/synthesize` only
  on click (no autoplay) and caches the result per message ID.

The design choice throughout was to treat voice as a translation layer at the
UI boundary — audio in, text out on one side; text in, audio out on the
other — rather than a parallel pipeline, so the RAG/agent/ticket logic that
was already working couldn't be disturbed by adding it.

## Completion Checklist

- [x] RAG answers use the supplied documents and return source names.
- [x] Unknown answers are not fabricated.
- [x] Ticket details are collected over multiple turns.
- [x] Tickets are created only after validation.
- [x] Repeated requests in one session do not create duplicate tickets.
- [x] API and Streamlit failures are displayed clearly.
- [x] Automated tests cover the principal success and failure paths, including the voice addition.
- [x] Voice input/output added without altering the original `/chat` contract or agent pipeline.