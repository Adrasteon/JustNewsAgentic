import pytest
from agents.critic import tools

def test_critique_synthesis_stub(monkeypatch):
    # Patch get_llama_model and pipeline to avoid heavy model loading
    monkeypatch.setattr(tools, "get_llama_model", lambda: (None, None))
    monkeypatch.setattr(tools, "pipeline", lambda *a, **kw: lambda prompt, **opts: [{"generated_text": "Critique output."}])
    result = tools.critique_synthesis("summary", ["a1", "a2"])
    assert isinstance(result, str)
    assert "Critique" in result or "output" in result

def test_critique_neutrality_stub(monkeypatch):
    monkeypatch.setattr(tools, "get_llama_model", lambda: (None, None))
    monkeypatch.setattr(tools, "pipeline", lambda *a, **kw: lambda prompt, **opts: [{"generated_text": "Neutrality critique."}])
    result = tools.critique_neutrality("original", "neutralized")
    assert isinstance(result, str)
    assert "Neutrality" in result or "critique" in result

def test_log_feedback(tmp_path, monkeypatch):
    log_file = tmp_path / "feedback.log"
    monkeypatch.setattr(tools, "FEEDBACK_LOG", str(log_file))
    tools.log_feedback("test_event", {"foo": "bar"})
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert any("test_event" in line for line in lines)
