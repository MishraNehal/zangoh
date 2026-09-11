from pathlib import Path

import pytest


@pytest.fixture
def knowledge_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "knowledge_base"
