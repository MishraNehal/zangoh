from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.llm.client import build_chat_model
from src.llm.workflow import build_support_workflow
from src.models import ChatResponse
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


class SupportPipeline:
    """Top-level binding for model, RAG, workflow, sessions, and ticket tool."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.model = build_chat_model(settings)
        self.retriever = KnowledgeRetriever(settings, documents_dir)
        self.sessions = SessionStore()
        self.tickets = TicketRepository()
        self.workflow = None
        self.ready = False

    async def initialize(self) -> None:
        """Initialize shared components once during FastAPI startup."""
        await self.retriever.initialize()
        self.workflow = build_support_workflow(
            self.model, self.retriever, self.sessions, self.tickets
        )
        self.ready = True

    async def process(self, session_id: str, message: str) -> ChatResponse:
        if not self.ready or self.workflow is None:
            raise ComponentNotReadyError("Support pipeline is not ready")

        session_id = (session_id or "").strip()
        message = (message or "").strip()
        if not session_id or not message:
            raise AgentProcessingError("session_id and message must not be blank")

        session = self.sessions.get_or_create(session_id)
        session.history.append({"role": "user", "content": message})

        initial_state = {
            "session_id": session_id,
            "customer_message": message,
        }

        try:
            result = await self.workflow.ainvoke(initial_state)
        except AgentProcessingError:
            raise
        except ComponentNotReadyError:
            raise
        except Exception as exc:
            raise AgentProcessingError(f"Failed to process message: {exc}") from exc

        response_text = result.get("response_text")
        if not response_text:
            raise AgentProcessingError("Workflow returned an empty response")

        ticket_id = result.get("ticket_id")
        if ticket_id and self.tickets.get(ticket_id) is None:
            ticket_id = None

        sources = result.get("sources") or []

        session.history.append({"role": "assistant", "content": response_text})

        return ChatResponse(
            success=True,
            session_id=session_id,
            response=response_text,
            sources=sources,
            ticket_id=ticket_id,
        )