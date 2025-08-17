from agents.analyst import tools

def test_log_feedback(tmp_path, monkeypatch):
    log_file = tmp_path / "feedback.log"
    monkeypatch.setattr(tools, "FEEDBACK_LOG", str(log_file))
    tools.log_feedback("test_event", {"foo": "bar"})
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert any("test_event" in line for line in lines)

def test_score_bias_stub(monkeypatch):
    monkeypatch.setattr(tools, "get_mistral_model", lambda: (None, None))
    monkeypatch.setattr(tools, "pipeline", lambda *a, **kw: lambda prompt, **opts: [{"score": 0.7}])
    score = tools.score_bias("Some text")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_score_sentiment_stub(monkeypatch):
    monkeypatch.setattr(tools, "get_mistral_model", lambda: (None, None))
    monkeypatch.setattr(tools, "pipeline", lambda *a, **kw: lambda prompt, **opts: [{"label": "positive"}])
    score = tools.score_sentiment("Some text")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_identify_entities_stub(monkeypatch):
    monkeypatch.setattr(tools, "get_mistral_model", lambda: (None, None))
    monkeypatch.setattr(tools, "pipeline", lambda *a, **kw: lambda prompt, **opts: [{"word": "NASA", "entity_group": "ORG"}])
    entities = tools.identify_entities("NASA launches Artemis.")
    assert isinstance(entities, list)
    assert "NASA" in entities
