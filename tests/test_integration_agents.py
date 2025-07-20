import requests
import time

BASE_URLS = {
    'mcp_bus': 'http://localhost:8000',
    'chief_editor': 'http://localhost:8001',
    'scout': 'http://localhost:8002',
    'fact_checker': 'http://localhost:8003',
    'analyst': 'http://localhost:8004',
    'synthesizer': 'http://localhost:8005',
    'critic': 'http://localhost:8006',
    'memory': 'http://localhost:8007',
}

def wait_for_service(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Service at {url} did not become available in time.")

def test_services_up():
    for name, url in BASE_URLS.items():
        wait_for_service(url)

def test_chief_editor_brief():
    url = BASE_URLS['chief_editor'] + '/request_story_brief'
    payload = {"args": ["Artemis program", "latest developments"], "kwargs": {}}
    r = requests.post(url, json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "brief requested"

def test_scout_discover_sources():
    url = BASE_URLS['scout'] + '/discover_sources'
    payload = {"args": ["Artemis program"], "kwargs": {}}
    r = requests.post(url, json=payload)
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_fact_checker_validate():
    url = BASE_URLS['fact_checker'] + '/validate_is_news'
    payload = {"args": ["NASA announced a new Artemis mission in July 2025 in Houston, Texas. The mission will..."], "kwargs": {}}
    r = requests.post(url, json=payload)
    assert r.status_code == 200
    assert isinstance(r.json(), bool)

def test_analyst_score_bias():
    url = BASE_URLS['analyst'] + '/score_bias'
    payload = {"args": ["This is clearly a great success for NASA."], "kwargs": {}}
    r = requests.post(url, json=payload)
    assert r.status_code == 200
    assert isinstance(r.json(), float)

def test_synthesizer_cluster():
    url = BASE_URLS['synthesizer'] + '/cluster_articles'
    payload = {"args": [["a1", "a2"]], "kwargs": {}}
    r = requests.post(url, json=payload)
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_critic_critiques():
    url = BASE_URLS['critic'] + '/critique_synthesis'
    payload = {"args": ["Short summary.", ["a1", "a2"]], "kwargs": {}}
    r = requests.post(url, json=payload)
    assert r.status_code == 200
    assert isinstance(r.json(), str)

def test_memory_save_and_get_article():
    # Save article
    url = BASE_URLS['memory'] + '/save_article'
    payload = {"content": "NASA Artemis mission details...", "metadata": {"author": "Reporter"}}
    r = requests.post(url, json=payload)
    assert r.status_code == 200
    article_id = r.json()["id"]
    # Get article
    url = BASE_URLS['memory'] + f'/get_article/{article_id}'
    r = requests.get(url)
    assert r.status_code == 200
    assert r.json()["content"].startswith("NASA Artemis mission")
