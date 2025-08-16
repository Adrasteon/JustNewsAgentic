import asyncio
import pytest

# Global fixtures for tests

@pytest.fixture(scope="session")
def event_loop():
    """Create an asyncio event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def dummy_model(monkeypatch):
    """Return a simple deterministic dummy model for unit tests.

    Usage:
        monkeypatch.setattr('agents.newsreader.model_loader.load_model', lambda *a, **k: DummyModel())
    """

    class DummyModel:
        def predict(self, text):
            return {"label": "dummy", "score": 1.0}

        async def apredict(self, text):
            return {"label": "dummy_async", "score": 1.0}

    return DummyModel()
