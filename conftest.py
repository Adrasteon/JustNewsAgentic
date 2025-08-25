import pytest


@pytest.fixture
def articles():
    # Shared sample articles used across the test suite
    return [
        "NASA launches a new mission to the Moon.",
        "Economic outlook improves as markets recover.",
        "Local community rallies to support small businesses."
    ]


class DummyModel:
    def predict(self, texts):
        # If a single string is provided, return a dict-like payload expected by some tests
        if isinstance(texts, str):
            return {"label": "dummy", "score": 0.5}
        # For list inputs, return a list of floats as a simple batch response
        if isinstance(texts, (list, tuple)):
            return [0.5 for _ in texts]
        return {"label": "dummy", "score": 0.5}


@pytest.fixture
def dummy_model():
    return DummyModel()
