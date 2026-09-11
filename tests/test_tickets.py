import pytest
from pydantic import ValidationError

from src.models import TicketCreate
from src.tools.ticket_tool import TicketRepository, create_ticket_tool


def _sample_request(**overrides) -> TicketCreate:
    defaults = dict(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Possible duplicate payment charge",
    )
    defaults.update(overrides)
    return TicketCreate(**defaults)


def test_repository_prevents_duplicate_ticket_per_session() -> None:
    repository = TicketRepository()
    request = _sample_request()

    first = repository.create("session-1", request)
    second = repository.create("session-1", request)

    assert first.ticket_id == second.ticket_id
    assert len(list(repository.all())) == 1


def test_unique_ticket_ids_across_different_sessions() -> None:
    repository = TicketRepository()

    first = repository.create("session-1", _sample_request())
    second = repository.create("session-2", _sample_request())

    assert first.ticket_id != second.ticket_id
    assert len(list(repository.all())) == 2


def test_ticket_preserves_validated_category_and_customer_details() -> None:
    repository = TicketRepository()
    request = _sample_request(
        customer_name="Rahul Verma",
        customer_email="rahul@example.com",
        category="technical",
        summary="App crashes on login",
    )

    ticket = repository.create("session-1", request)

    assert ticket.customer_name == "Rahul Verma"
    assert ticket.customer_email == "rahul@example.com"
    assert ticket.category == "technical"
    assert ticket.status == "open"
    assert ticket.ticket_id.startswith("CST-")
    assert ticket.created_at is not None


def test_get_returns_existing_ticket_by_id() -> None:
    repository = TicketRepository()
    created = repository.create("session-1", _sample_request())

    fetched = repository.get(created.ticket_id)

    assert fetched is not None
    assert fetched.ticket_id == created.ticket_id


def test_get_returns_none_for_unknown_ticket_id() -> None:
    repository = TicketRepository()
    assert repository.get("CST-2026-9999") is None


def test_ticket_create_rejects_invalid_email() -> None:
    with pytest.raises(ValidationError):
        _sample_request(customer_email="not-an-email")


def test_ticket_create_rejects_unsupported_category() -> None:
    with pytest.raises(ValidationError):
        _sample_request(category="shipping")


def test_ticket_tool_invokes_repository_and_returns_id() -> None:
    repository = TicketRepository()
    tool = create_ticket_tool(repository, "session-1")

    ticket_id = tool.invoke(
        {
            "customer_name": "Asha",
            "customer_email": "asha@example.com",
            "issue_description": "Order never arrived",
            "category": "order",
            "summary": "Missing order",
        }
    )

    assert ticket_id == repository.get(ticket_id).ticket_id


def test_ticket_tool_retry_returns_first_ticket_not_a_new_one() -> None:
    repository = TicketRepository()
    tool = create_ticket_tool(repository, "session-1")
    payload = {
        "customer_name": "Asha",
        "customer_email": "asha@example.com",
        "issue_description": "Order never arrived",
        "category": "order",
        "summary": "Missing order",
    }

    first_id = tool.invoke(payload)
    second_id = tool.invoke(payload)

    assert first_id == second_id
    assert len(list(repository.all())) == 1