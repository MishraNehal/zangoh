import os
import uuid

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
# Speech synthesis/transcription can take longer than a typical /chat call.
VOICE_REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0)

st.set_page_config(page_title="Customer Support", page_icon="🎧")
st.title("Customer Support")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
# --- voice-feature state (additive; nothing above this line changed) -------
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}  # message_id -> (audio_bytes, media_type)
if "synthesizing_id" not in st.session_state:
    st.session_state.synthesizing_id = None
if "last_recording_id" not in st.session_state:
    st.session_state.last_recording_id = None
if "pending_transcript" not in st.session_state:
    st.session_state.pending_transcript = None

with st.sidebar:
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    if st.button("Start new conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.audio_cache = {}
        st.session_state.synthesizing_id = None
        st.session_state.last_recording_id = None
        st.session_state.pending_transcript = None
        st.rerun()


def transcribe_audio(audio_file) -> dict:
    """POST recorded audio to /voice/transcribe. Never raises; always
    returns a dict with either a transcript or an error message."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/voice/transcribe",
            files={
                "audio": (
                    getattr(audio_file, "name", "recording.wav") or "recording.wav",
                    audio_file.getvalue(),
                    getattr(audio_file, "type", None) or "audio/wav",
                )
            },
            timeout=VOICE_REQUEST_TIMEOUT,
        )
    except httpx.ConnectError:
        return {"success": False, "error": "Could not reach the support API. Is the backend running?"}
    except httpx.TimeoutException:
        return {"success": False, "error": "Transcription took too long. Please try again."}

    try:
        data = response.json()
    except ValueError:
        return {"success": False, "error": "Received an unreadable response from the server."}

    if response.status_code >= 400 or not data.get("success", True):
        return {"success": False, "error": data.get("error", f"Transcription failed ({response.status_code}).")}
    return data


def synthesize_audio(message_id: str, text: str) -> tuple[bytes | None, str | None, str | None]:
    """POST response text to /voice/synthesize. Returns (audio_bytes,
    media_type, error) -- exactly one of (audio_bytes, error) is set."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/voice/synthesize",
            json={"message_id": message_id, "text": text},
            timeout=VOICE_REQUEST_TIMEOUT,
        )
    except httpx.ConnectError:
        return None, None, "Could not reach the support API. Is the backend running?"
    except httpx.TimeoutException:
        return None, None, "Speech synthesis took too long. Please try again."

    content_type = response.headers.get("content-type", "")
    if response.status_code >= 400 or content_type.startswith("application/json"):
        try:
            error_detail = response.json().get("error", "Speech synthesis failed.")
        except ValueError:
            error_detail = f"Speech synthesis failed ({response.status_code})."
        return None, None, error_detail

    return response.content, content_type, None


def render_speaker_control(message: dict) -> None:
    message_id = message["id"]
    cache = st.session_state.audio_cache
    already_generated = message_id in cache
    is_this_one_busy = st.session_state.synthesizing_id == message_id
    # Disable every speaker button while any one of them is mid-synthesis, so
    # a rapid extra click can't fire a second, overlapping request for either
    # the same message or a different one.
    any_busy = st.session_state.synthesizing_id is not None

    label = "🔊 Generating..." if is_this_one_busy else ("🔁 Replay" if already_generated else "🔊 Play response")
    clicked = st.button(label, key=f"speak_{message_id}", disabled=any_busy)

    if clicked and not already_generated:
        st.session_state.synthesizing_id = message_id
        with st.spinner("Generating audio..."):
            audio_bytes, media_type, error = synthesize_audio(message_id, message["content"])
        st.session_state.synthesizing_id = None
        if error:
            st.error(f"Couldn't generate audio for this response: {error}")
        else:
            cache[message_id] = (audio_bytes, media_type)
        st.rerun()

    if already_generated:
        audio_bytes, media_type = cache[message_id]
        st.audio(audio_bytes, format=media_type)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))
        if message.get("ticket_id"):
            st.success(f"Ticket created: {message['ticket_id']}")
        if message["role"] == "assistant":
            render_speaker_control(message)


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
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": data.get("response", ""),
                "sources": data.get("sources") or [],
                "ticket_id": data.get("ticket_id"),
            }
            st.session_state.messages.append(assistant_message)
            render_message(assistant_message)


# --- Voice input: mic -> STT -> editable transcript -> the SAME /chat path -
# This never bypasses send_message(); it only produces the text that a typed
# message would otherwise have supplied.
with st.expander("🎤 Or record your message", expanded=False):
    audio_value = st.audio_input("Record")

    if audio_value is not None:
        recording_id = getattr(audio_value, "file_id", None) or id(audio_value)
        if recording_id != st.session_state.last_recording_id:
            st.session_state.last_recording_id = recording_id
            with st.spinner("Transcribing..."):
                result = transcribe_audio(audio_value)
            if result.get("success", True) and result.get("transcript"):
                st.session_state.pending_transcript = result["transcript"]
            else:
                st.session_state.pending_transcript = None
                st.error(result.get("error", "Could not understand that recording. Please try again."))

    if st.session_state.pending_transcript is not None:
        edited_transcript = st.text_area(
            "Review and edit the transcript before sending",
            value=st.session_state.pending_transcript,
            key="transcript_editor",
        )
        col_submit, col_discard = st.columns(2)
        with col_submit:
            submit_clicked = st.button("Submit transcript", type="primary")
        with col_discard:
            discard_clicked = st.button("Discard")

        if submit_clicked:
            text_to_send = edited_transcript.strip()
            st.session_state.pending_transcript = None
            st.session_state.last_recording_id = None
            if text_to_send:
                st.session_state.messages.append(
                    {"id": str(uuid.uuid4()), "role": "user", "content": text_to_send}
                )
                with st.chat_message("user"):
                    st.write(text_to_send)
                send_message(text_to_send)
            st.rerun()
        elif discard_clicked:
            st.session_state.pending_transcript = None
            st.session_state.last_recording_id = None
            st.rerun()


if prompt := st.chat_input("How can we help?"):
    st.session_state.messages.append({"id": str(uuid.uuid4()), "role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    send_message(prompt)
