from src.models import TicketCreate
from src.tools.ticket_tool import TicketRepository


def test_repository_prevents_duplicate_ticket_per_session() -> None:
    # The repository is provided working code. This test is also an integration
    # contract for the eventual agent tool: one session must produce one ticket.
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Possible duplicate payment charge",
    )

    first = repository.create("session-1", request)
    second = repository.create("session-1", request)

    assert first.ticket_id == second.ticket_id
    assert len(list(repository.all())) == 1


# CANDIDATE TEST PLAN
#
# Add coverage for:
# - Unique IDs across different sessions.
# - Exact preservation of validated category and customer details.
# - Retrieval of an existing ticket through the repository and API.
# - A 404 response for an unknown ticket ID.
# - Agent retries that return the first ticket rather than claiming a new one.
# - Tool/model failure paths that do not leave a false ticket ID in session.
