import os
import uuid

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)

st.set_page_config(page_title="Customer Support", page_icon="🎧")
st.title("Customer Support")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    if st.button("Start new conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))
        if message.get("ticket_id"):
            st.success(f"Ticket created: {message['ticket_id']}")


for message in st.session_state.messages:
    render_message(message)


def send_message(text: str) -> None:
    payload = {"session_id": st.session_state.session_id, "message": text}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/chat", json=payload, timeout=REQUEST_TIMEOUT
                )
            except httpx.ConnectError:
                st.error("Could not reach the support API. Is the backend running?")
                return
            except httpx.TimeoutException:
                st.error("The support agent took too long to respond. Please try again.")
                return

            if response.status_code == 422:
                st.error(f"Invalid request: {response.text}")
                return
            if response.status_code == 503:
                st.error("A required component isn't ready yet. Please try again shortly.")
                return
            if response.status_code >= 400:
                st.error(f"The support agent hit an error ({response.status_code}).")
                return

            try:
                data = response.json()
            except ValueError:
                st.error("Received an unreadable response from the server.")
                return

            assistant_message = {
                "role": "assistant",
                "content": data.get("response", ""),
                "sources": data.get("sources") or [],
                "ticket_id": data.get("ticket_id"),
            }
            st.session_state.messages.append(assistant_message)
            render_message(assistant_message)


if prompt := st.chat_input("How can we help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    send_message(prompt)