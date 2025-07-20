from agents.synthesizer import tools

def test_cluster_articles_kmeans():
    articles = ["NASA launches Artemis mission.", "SpaceX launches Starlink satellites.", "NASA plans Mars mission."]
    clusters = tools.cluster_articles(articles, n_clusters=2)
    assert isinstance(clusters, list)
    assert all(isinstance(c, list) for c in clusters)
    # All indices should be present
    all_indices = [i for cluster in clusters for i in cluster]
    assert set(all_indices) == set(range(len(articles)))

def test_neutralize_text_stub(monkeypatch):
    # Patch get_llama_model and pipeline to avoid heavy model loading
    monkeypatch.setattr(tools, "get_llama_model", lambda: (None, None))
    monkeypatch.setattr(tools, "pipeline", lambda *a, **kw: lambda prompt, **opts: [{"generated_text": "Neutralized text."}])
    result = tools.neutralize_text("This is a BIASED and STRONG statement!")
    assert isinstance(result, str)
    assert "Neutralized" in result

def test_aggregate_cluster_stub(monkeypatch):
    monkeypatch.setattr(tools, "get_llama_model", lambda: (None, None))
    monkeypatch.setattr(tools, "pipeline", lambda *a, **kw: lambda prompt, **opts: [{"generated_text": "Summary."}])
    articles = ["NASA launches Artemis mission.", "SpaceX launches Starlink satellites."]
    result = tools.aggregate_cluster(articles)
    assert isinstance(result, str)
    assert "Summary" in result

def test_log_feedback(tmp_path, monkeypatch):
    log_file = tmp_path / "feedback.log"
    monkeypatch.setattr(tools, "FEEDBACK_LOG", str(log_file))
    tools.log_feedback("test_event", {"foo": "bar"})
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert any("test_event" in line for line in lines)
