import pytest
from agents.memory import tools

def test_log_feedback(tmp_path, monkeypatch):
    log_file = tmp_path / "feedback.log"
    monkeypatch.setattr(tools, "FEEDBACK_LOG", str(log_file))
    tools.log_feedback("test_event", {"foo": "bar"})
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert any("test_event" in line for line in lines)

def test_save_article_stub():
    result = tools.save_article("Test content", {"author": "A"})
    assert isinstance(result, dict)
    assert "id" in result

def test_get_article_stub():
    result = tools.get_article(123)
    assert isinstance(result, dict)
    assert result["id"] == 123

def test_vector_search_articles_stub():
    result = tools.vector_search_articles("query", top_k=2)
    assert isinstance(result, list)

def test_log_training_example_stub():
    result = tools.log_training_example("task", {"x": 1}, {"y": 2}, "critique")
    assert isinstance(result, dict)
    assert result["status"] == "logged"
