import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MCP_BUS_URL = "http://localhost:8000"
AGENTS = {
    "analyst": "http://localhost:8004",
    "critic": "http://localhost:8003",
    "synthesizer": "http://localhost:8005",
}
TOOLS = {
    "analyst": ["score_bias", "score_sentiment", "identify_entities"],
    "critic": ["critique_synthesis", "critique_neutrality"],
    "synthesizer": ["cluster_articles", "neutralize_text", "aggregate_cluster"],
}

def test_agent_registration():
    logger.info("Testing agent registration...")
    response = requests.get(f"{MCP_BUS_URL}/agents")
    assert response.status_code == 200, "Failed to fetch registered agents"
    registered_agents = response.json()
    for agent, url in AGENTS.items():
        assert agent in registered_agents, f"Agent {agent} not registered"
        assert registered_agents[agent] == url, f"Agent {agent} has incorrect URL"
    logger.info("Agent registration test passed.")

def test_tool_endpoints():
    logger.info("Testing tool endpoints...")
    for agent, tools in TOOLS.items():
        for tool in tools:
            url = f"{AGENTS[agent]}/{tool}"
            logger.info(f"Testing endpoint: {url}")
            try:
                response = requests.post(url, json={"args": [], "kwargs": {}})
                assert response.status_code == 200, f"Tool {tool} for agent {agent} failed"
                logger.info(f"Tool {tool} for agent {agent} passed.")
            except Exception as e:
                logger.error(f"Error testing tool {tool} for agent {agent}: {e}")

def test_mcp_call():
    logger.info("Testing MCP Bus /call endpoint...")
    for agent, tools in TOOLS.items():
        for tool in tools:
            payload = {
                "agent": agent,
                "tool": tool,
                "args": [],
                "kwargs": {}
            }
            response = requests.post(f"{MCP_BUS_URL}/call", json=payload)
            assert response.status_code == 200, f"MCP Bus call to {tool} for agent {agent} failed"
            logger.info(f"MCP Bus call to {tool} for agent {agent} passed.")

if __name__ == "__main__":
    test_agent_registration()
    test_tool_endpoints()
    test_mcp_call()
