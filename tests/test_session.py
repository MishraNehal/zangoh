from src.sessions.store import ConversationState


def test_session_reports_only_missing_ticket_fields() -> None:
    # This supplied test establishes the expected collection order. Add tests for
    # an empty state, a complete state, and updates made across multiple turns.
    state = ConversationState(customer_name="Asha", customer_email="asha@example.com")

    assert state.missing_ticket_fields() == ["issue_description", "category"]


# CANDIDATE TEST PLAN
#
# Add focused tests rather than relying only on the five-minute demonstration:
# - A new session does not reuse another customer's values.
# - Fields already collected are not requested again.
# - Invalid email input remains uncommitted and produces a useful follow-up.
# - Unsupported categories are rejected or mapped only after explicit logic.
# - Conversation history preserves role and content in the correct order.
# - A completed session retains the repository-issued ticket ID.
