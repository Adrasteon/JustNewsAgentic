
import pytest
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

@pytest.mark.skip(reason="mcp_bus not started in CI")
def test_services_up():
    pass

@pytest.mark.skip(reason="chief_editor not started in CI")
def test_chief_editor_brief():
    pass

@pytest.mark.skip(reason="scout not started in CI")
def test_scout_discover_sources():
    pass

@pytest.mark.skip(reason="fact_checker not started in CI")
def test_fact_checker_validate():
    pass

@pytest.mark.skip(reason="analyst not started in CI")
def test_analyst_score_bias():
    pass

@pytest.mark.skip(reason="synthesizer not started in CI")
def test_synthesizer_cluster():
    pass

@pytest.mark.skip(reason="critic not started in CI")
def test_critic_critiques():
    pass

@pytest.mark.skip(reason="memory not started in CI")
def test_memory_save_and_get_article():
    pass
