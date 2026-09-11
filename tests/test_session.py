from src.sessions.store import ConversationState, SessionStore


def test_session_reports_only_missing_ticket_fields() -> None:
    state = ConversationState(customer_name="Asha", customer_email="asha@example.com")
    assert state.missing_ticket_fields() == ["issue_description", "category"]


def test_empty_state_reports_all_fields_missing() -> None:
    state = ConversationState()
    assert state.missing_ticket_fields() == [
        "customer_name",
        "customer_email",
        "issue_description",
        "category",
    ]


def test_complete_state_reports_nothing_missing() -> None:
    state = ConversationState(
        customer_name="Asha",
        customer_email="asha@example.com",
        issue_description="Payment charged twice",
        category="payment",
    )
    assert state.missing_ticket_fields() == []


def test_fields_fill_in_across_multiple_turns() -> None:
    state = ConversationState()
    state.customer_name = "Asha"
    assert state.missing_ticket_fields() == [
        "customer_email",
        "issue_description",
        "category",
    ]
    state.customer_email = "asha@example.com"
    state.issue_description = "Payment charged twice"
    assert state.missing_ticket_fields() == ["category"]
    state.category = "payment"
    assert state.missing_ticket_fields() == []


def test_new_session_does_not_reuse_another_customers_values() -> None:
    store = SessionStore()
    first = store.get_or_create("session-a")
    first.customer_name = "Asha"

    second = store.get_or_create("session-b")
    assert second.customer_name is None
    assert second is not first


def test_get_or_create_returns_same_state_for_repeated_session_id() -> None:
    store = SessionStore()
    first_call = store.get_or_create("session-a")
    first_call.customer_name = "Asha"

    second_call = store.get_or_create("session-a")
    assert second_call.customer_name == "Asha"
    assert second_call is first_call


def test_get_or_create_rejects_blank_session_id() -> None:
    store = SessionStore()
    import pytest

    with pytest.raises(ValueError):
        store.get_or_create("   ")


def test_history_preserves_role_and_content_order() -> None:
    state = ConversationState()
    state.history.append({"role": "user", "content": "My payment was charged twice"})
    state.history.append({"role": "assistant", "content": "I'm sorry to hear that."})

    assert [turn["role"] for turn in state.history] == ["user", "assistant"]
    assert state.history[0]["content"] == "My payment was charged twice"


def test_completed_session_retains_repository_issued_ticket_id() -> None:
    state = ConversationState(
        customer_name="Asha",
        customer_email="asha@example.com",
        issue_description="Payment charged twice",
        category="payment",
    )
    state.ticket_id = "CST-2026-0001"
    assert state.missing_ticket_fields() == []
    assert state.ticket_id == "CST-2026-0001"