from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ValidationError

from src.llm.prompts import (
    ANSWER_TEMPLATE,
    DECIDE_SYSTEM_PROMPT,
    DECIDE_TEMPLATE,
    FOLLOWUP_QUESTIONS,
    SYSTEM_PROMPT,
)
from src.models import TicketCreate
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import ConversationState, SessionStore
from src.tools.ticket_tool import TicketRepository, create_ticket_tool
from src.utils.errors import AgentProcessingError


class SupportWorkflowState(TypedDict, total=False):
    """Typed state shared by the supplied LangGraph node skeletons."""

    session_id: str
    customer_message: str
    messages: Annotated[list, add_messages]
    retrieved_chunks: list[dict[str, str]]
    route: str
    extracted_fields: dict[str, str]
    response_text: str
    sources: list[str]
    ticket_id: str | None


class SupportDecision(BaseModel):
    """Structured output contract for the decide node."""

    route: Literal["answer", "ticket"]
    customer_name: str | None = None
    customer_email: str | None = None
    issue_description: str | None = None
    category: Literal["order", "payment", "account", "technical", "other"] | None = None


def _format_session(session: ConversationState) -> str:
    fields = {
        "customer_name": session.customer_name,
        "customer_email": session.customer_email,
        "issue_description": session.issue_description,
        "category": session.category,
    }
    known = {k: v for k, v in fields.items() if v}
    if not known:
        return "No ticket details collected yet."
    return "\n".join(f"- {k}: {v}" for k, v in known.items())


def build_support_workflow(
    model: BaseChatModel,
    retriever: KnowledgeRetriever,
    sessions: SessionStore,
    tickets: TicketRepository,
):
    """Build and compile the agent graph, wired to its runtime dependencies."""

    decision_model = model.with_structured_output(SupportDecision, method="json_schema")

    async def retrieve(state: SupportWorkflowState) -> SupportWorkflowState:
        message = state.get("customer_message", "")
        chunks = await retriever.search(message)

        seen: set[tuple[str, str]] = set()
        deduped: list[dict[str, str]] = []
        for chunk in chunks:
            key = (chunk.get("source", ""), chunk.get("content", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)

        return {"retrieved_chunks": deduped}

    async def decide(state: SupportWorkflowState) -> SupportWorkflowState:
        session = sessions.get_or_create(state["session_id"])
        context = "\n\n".join(
            f"[{c['source']}] {c['content']}" for c in state.get("retrieved_chunks", [])
        ) or "No matching knowledge base content."

        prompt = DECIDE_TEMPLATE.format(
            context=context,
            session=_format_session(session),
            message=state.get("customer_message", ""),
        )

        try:
            decision = await decision_model.ainvoke(
                [SystemMessage(content=DECIDE_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
        except ValidationError as exc:
            raise AgentProcessingError(f"Could not interpret the request: {exc}") from exc

        extracted = {
            key: value
            for key, value in decision.model_dump().items()
            if key != "route" and value
        }

        return {"route": decision.route, "extracted_fields": extracted}

    async def answer(state: SupportWorkflowState) -> SupportWorkflowState:
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {
                "response_text": (
                    "I don't have information about that in our knowledge base. "
                    "I can raise a support ticket for you if you'd like — just let me know."
                ),
                "sources": [],
            }

        context = "\n\n".join(f"[{c['source']}] {c['content']}" for c in chunks)
        session = sessions.get_or_create(state["session_id"])
        prompt = ANSWER_TEMPLATE.format(
            context=context,
            session=_format_session(session),
            message=state.get("customer_message", ""),
        )

        try:
            result = await model.ainvoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
        except Exception as exc:
            raise AgentProcessingError(f"Failed to generate an answer: {exc}") from exc

        sources = sorted({c["source"] for c in chunks})
        return {"response_text": result.content, "sources": sources}

    async def collect_or_create(state: SupportWorkflowState) -> SupportWorkflowState:
        session_id = state["session_id"]
        session = sessions.get_or_create(session_id)

        for field in ("customer_name", "customer_email", "issue_description", "category"):
            value = state.get("extracted_fields", {}).get(field)
            if value:
                setattr(session, field, value)

        if session.ticket_id:
            return {
                "response_text": (
                    f"You already have an open ticket ({session.ticket_id}) for this issue. "
                    "Our team will follow up by email."
                ),
                "sources": [],
                "ticket_id": session.ticket_id,
            }

        missing = session.missing_ticket_fields()
        if missing:
            question = FOLLOWUP_QUESTIONS.get(missing[0], "Could you share a bit more detail?")
            return {"response_text": question, "sources": [], "ticket_id": None}

        summary_source = session.issue_description or ""
        summary = summary_source if len(summary_source) <= 160 else summary_source[:157] + "..."
        if len(summary) < 5:
            summary = (summary + " - support request").strip()[:160]

        try:
            ticket_data = TicketCreate(
                customer_name=session.customer_name or "",
                customer_email=session.customer_email or "",
                issue_description=session.issue_description or "",
                category=session.category,  # type: ignore[arg-type]
                summary=summary,
            )
        except ValidationError:
            session.customer_email = None
            return {
                "response_text": "That email address doesn't look valid. Could you share the correct one?",
                "sources": [],
                "ticket_id": None,
            }

        tool = create_ticket_tool(tickets, session_id)
        try:
            ticket_id = tool.invoke(ticket_data.model_dump())
        except Exception as exc:
            raise AgentProcessingError(f"Failed to create ticket: {exc}") from exc

        session.ticket_id = ticket_id
        return {
            "response_text": (
                f"Thanks — I've created ticket {ticket_id} for you. "
                "Our team will follow up at the email you provided."
            ),
            "sources": [],
            "ticket_id": ticket_id,
        }

    def select_route(state: SupportWorkflowState) -> str:
        route = state.get("route")
        if route not in ("answer", "ticket"):
            raise AgentProcessingError(f"Workflow produced an unknown route: {route!r}")
        return route

    graph = StateGraph(SupportWorkflowState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("decide", decide)
    graph.add_node("answer", answer)
    graph.add_node("ticket", collect_or_create)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges(
        "decide",
        select_route,
        {"answer": "answer", "ticket": "ticket"},
    )
    graph.add_edge("answer", END)
    graph.add_edge("ticket", END)
    return graph.compile()