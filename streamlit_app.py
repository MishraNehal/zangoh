import os
import uuid

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Support", page_icon="🎧")
st.title("Customer Support")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # TODO: For completed assistant messages, also restore any source names
        # and ticket ID saved with this message. Rendering must survive ordinary
        # Streamlit reruns instead of depending on temporary local variables.

if prompt := st.chat_input("How can we help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # TODO: Complete the UI-to-API flow:
    #
    # 1. Build JSON containing the stable ``session_id`` and current ``prompt``.
    # 2. Call ``POST {API_BASE_URL}/chat`` with a finite timeout.
    # 3. Show a spinner while waiting, without accepting duplicate submission.
    # 4. Distinguish HTTP errors, connection failures, invalid response JSON,
    #    and valid agent responses.
    # 5. Append the assistant result to ``st.session_state.messages`` with its
    #    response text, sources, and optional ticket ID.
    # 6. Render source names separately from the answer. Do not expose absolute
    #    paths returned accidentally by an implementation.
    # 7. Render a clear ticket-success element only when ``ticket_id`` is set.
    # 8. Keep earlier chat history visible if the latest request fails.
    #
    # Do not call the model, vector store, or TicketRepository directly from
    # Streamlit. FastAPI remains the single backend entry point.
    #
    # RESPONSE SHAPE
    # - Read ``response`` as the assistant-visible text.
    # - Treat ``sources`` as an optional list of filenames.
    # - Treat ``ticket_id`` as optional; do not manufacture a placeholder ID.
    # - Retain all three fields in the stored assistant message for rerenders.
    #
    # SESSION RULES
    # - Generate the UUID once when the browser session begins.
    # - Send the same UUID on every typed and, later, transcribed request.
    # - A Streamlit rerun must not reset chat history or the session ID.
    # - A deliberate new-conversation control may reset both together.
    #
    # FAILURE RULES
    # - Use a finite connect/read timeout.
    # - A 422 response should display useful input-validation information.
    # - A 503 response should explain that a required component is unavailable.
    # - Malformed JSON must become a friendly UI error, not a traceback.
    # - Never append an unsuccessful placeholder as an assistant answer.
    # - Never discard previously rendered messages after the latest failure.
    #
    # MIDWAY COMPATIBILITY
    # - Keep message records extensible with a stable per-message identifier.
    # - The midway speaker action will attach audio to that assistant message.
    # - The midway transcript must enter through this same chat submission path.
    # - Do not embed STT/TTS logic into the base API request implementation.
    with st.chat_message("assistant"):
        st.info("Complete the API integration TODO.")
