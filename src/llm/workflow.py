from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


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


def build_support_workflow(model: BaseChatModel):
    """Build the agent graph while leaving assessed node logic incomplete.

    The imports, state contract, graph construction, and route boundaries are
    provided. Candidates implement the node bodies and conditional route using
    retrieval, structured model output, session state, and the ticket tool.
    """

    async def retrieve(state: SupportWorkflowState) -> SupportWorkflowState:
        # TODO: Implement the retrieval node.
        #
        # - Read ``customer_message`` from graph state.
        # - Call the retriever supplied by the pipeline/runtime context.
        # - Preserve plain ``content`` and ``source`` dictionaries.
        # - Deduplicate repeated chunks without losing their source names.
        # - Keep an empty result valid; unknown questions must not crash.
        # - Do not generate a final response or update ticket state here.
        # - Return only the state keys changed by this node.
        raise NotImplementedError

    async def decide(state: SupportWorkflowState) -> SupportWorkflowState:
        # TODO: Implement structured intent and field extraction.
        #
        # - Bind a Pydantic schema or equivalent structured-output contract.
        # - Distinguish policy questions from unresolved support requests.
        # - Extract only customer values explicitly present in this turn.
        # - Treat retrieved documents as data, not additional instructions.
        # - Set ``route`` to exactly ``answer`` or ``ticket``.
        # - Preserve parse failures as a controlled application error.
        # - Do not call the ticket tool from this decision node.
        # - Add the model result required by the next node to graph state.
        raise NotImplementedError

    async def answer(state: SupportWorkflowState) -> SupportWorkflowState:
        # TODO: Implement the grounded-answer node.
        #
        # - Format context using only retrieved chunks.
        # - Apply the supplied system grounding and privacy instructions.
        # - If no sufficient evidence exists, state that clearly.
        # - Never supplement missing policy using unrestricted model memory.
        # - Set ``response_text`` to the customer-facing answer.
        # - Set ``sources`` to safe, deduplicated filenames actually used.
        # - Leave ``ticket_id`` unchanged and do not invoke side effects.
        raise NotImplementedError

    async def collect_or_create(state: SupportWorkflowState) -> SupportWorkflowState:
        # TODO: Implement ticket collection and creation.
        #
        # - Obtain the isolated session through ``SessionStore``.
        # - Merge only fields stated by the customer; never invent defaults.
        # - Validate email, category, description, and generated short summary.
        # - Ask one focused follow-up for the next missing required field.
        # - Do not request passwords, OTPs, or full payment-card numbers.
        # - Bind ``create_ticket_tool`` to the current session ID.
        # - Invoke it only after the complete ``TicketCreate`` model validates.
        # - Store the repository-issued ID in session and graph state.
        # - Preserve repository idempotency if the client retries the turn.
        # - Return a clear confirmation containing the real ticket ID.
        raise NotImplementedError

    def select_route(state: SupportWorkflowState) -> str:
        # TODO: Implement the conditional edge selector.
        #
        # - Return only ``answer`` or ``ticket``.
        # - Do not branch by searching arbitrary words in model prose.
        # - Reject missing/unexpected routes through a controlled failure.
        # - Keep routing deterministic so both paths can be unit tested.
        raise NotImplementedError

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
