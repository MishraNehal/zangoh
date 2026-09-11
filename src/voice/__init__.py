"""Voice enhancement: STT/TTS adapters wired into the existing text pipeline.

This package is intentionally decoupled from ``src.pipeline`` /
``src.llm.workflow``: it only ever produces or consumes plain text at its
boundary, so the existing agent, RAG, and ticket logic never need to know
voice input/output exists.
"""
