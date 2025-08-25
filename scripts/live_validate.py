#!/usr/bin/env python3
"""
Lightweight live validator for running a production-like end-to-end test.
- Checks /health endpoints for MCP bus and agents (defaults to common ports)
- Attempts a small number of best-effort POST/GET calls to discover working endpoints

This script intentionally avoids importing internal modules and instead talks to HTTP endpoints.
"""

from __future__ import annotations
import os
import sys
import time
import json
from typing import Dict, List, Any

try:
    import requests
except Exception:
    print("The 'requests' library is required. Install it in your environment (pip install requests).", file=sys.stderr)
    raise

ROOT = os.path.dirname(os.path.dirname(__file__))

AGENTS = {
    'mcp': int(os.environ.get('MCP_BUS_PORT', 8000)),
}


def discover_agents_from_mcp(timeout: int = 2) -> Dict[str, int]:
    """Query the MCP bus for registered agents and return a mapping agent->port.

    Falls back to static environment-based ports when MCP is unavailable.
    """
    agents: Dict[str, int] = {}
    mcp_port = int(os.environ.get('MCP_BUS_PORT', 8000))
    base = f"http://127.0.0.1:{mcp_port}"
    try:
        r = requests.get(base + '/agents', timeout=timeout)
        if r.status_code == 200:
            j = r.json()
            # j is expected to be a dict {name: "http://host:port"}
            from urllib.parse import urlparse

            for name, addr in j.items():
                try:
                    parsed = urlparse(addr)
                    port = int(parsed.port) if parsed.port else (80 if parsed.scheme == 'http' else 443)
                    agents[name] = port
                except Exception:
                    continue
    except Exception:
        # ignore discovery failures
        pass

    # If MCP discovery failed or returned a small set, supplement with env defaults
    env_defaults = {
        'memory': int(os.environ.get('MEMORY_AGENT_PORT', 8001)),
        'scout': int(os.environ.get('SCOUT_AGENT_PORT', 8002)),
        'fact_checker': int(os.environ.get('FACT_CHECKER_AGENT_PORT', 8003)),
        'analyst': int(os.environ.get('ANALYST_AGENT_PORT', 8004)),
        'synthesizer': int(os.environ.get('SYNTHESIZER_AGENT_PORT', 8005)),
        'critic': int(os.environ.get('CRITIC_AGENT_PORT', 8006)),
        'chief_editor': int(os.environ.get('CHIEF_EDITOR_AGENT_PORT', 8001)),
        'balancer': int(os.environ.get('BALANCER_PORT', 8009)),
    }
    for k, v in env_defaults.items():
        if k not in agents:
            agents[k] = v
    # Always ensure MCP bus itself is present
    agents['mcp'] = int(os.environ.get('MCP_BUS_PORT', 8000))
    return agents

# Per-agent functional probes. Each entry is a list of tuples: (method, path, payload_or_none)
FUNCTIONAL_PROBES = {
    'mcp': [
        ('GET', '/health', None),
    ],
    'memory': [
        ('GET', '/health', None),
        ('POST', '/tool', {'test': True}),
    ],
    'scout': [
        ('GET', '/health', None),
    ],
    'analyst': [
        ('GET', '/health', None),
        ('POST', '/predict', {'text': 'Hello world', 'test': True}),
    ],
    'synthesizer': [
        ('GET', '/health', None),
        ('POST', '/run', {'text': 'Summarize this', 'test': True}),
    ],
    'fact_checker': [
        ('GET', '/health', None),
    ],
    'critic': [
        ('GET', '/health', None),
    ],
    'chief_editor': [
        ('GET', '/health', None),
    ],
    'balancer': [
        ('GET', '/health', None),
    ],
}


def check_health(port: int, timeout: int = 5, retries: int = 3, backoff: float = 0.8) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    attempt = 0
    while attempt < retries:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return True
            # non-200: retry
            attempt += 1
            time.sleep(backoff)
        except Exception:
            attempt += 1
            time.sleep(backoff)
    return False


def discover_and_exercise(agent_name: str, port: int, retries: int = 2) -> Dict[str, Any]:
    base = f"http://127.0.0.1:{port}"
    res: Dict[str, Any] = {'agent': agent_name, 'port': port, 'health': 'unknown', 'probes': []}
    # Basic health
    try:
        h = requests.get(base + '/health', timeout=5)
        res['health'] = h.status_code
    except Exception as e:
        res['health'] = f'error:{e}'
        return res

    # Try to discover advertised probes via /info
    probes = []
    try:
        info = requests.get(base + '/info', timeout=2)
        if info.status_code == 200:
            j = info.json()
            if isinstance(j, dict) and 'probes' in j and isinstance(j['probes'], list):
                for p in j['probes']:
                    if isinstance(p, dict) and 'method' in p and 'path' in p:
                        probes.append((p['method'].upper(), p['path'], None))
    except Exception:
        # ignore discovery failures and fall back
        probes = []

    if not probes:
        probes = FUNCTIONAL_PROBES.get(agent_name, [])
    for method, path, payload in probes:
        url = base + path
        attempt = 0
        success = False
        last_status = None
        while attempt <= retries and not success:
            try:
                if method == 'GET':
                    r = requests.get(url, timeout=6)
                else:
                    r = requests.post(url, json=payload or {'test': True}, timeout=6)
                last_status = r.status_code
                if 200 <= r.status_code < 300:
                    success = True
                else:
                    # Treat non-2xx as a warning; record status but don't abort
                    attempt += 1
                    time.sleep(0.5)
            except Exception as e:
                last_status = f'error:{e}'
                attempt += 1
                time.sleep(0.5)
        res['probes'].append({'method': method, 'path': path, 'status': last_status, 'success': success})
    return res


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Live validator for agents')
    parser.add_argument('--output', '-o', help='Write JSON summary to file', default=None)
    args = parser.parse_args()

    print('Live validator starting...')
    results: List[Dict[str, Any]] = []
    agents_map = discover_agents_from_mcp()
    for name, port in agents_map.items():
        print(f'Checking {name} on port {port}...')
        ok = check_health(port)
        if not ok:
            print(f'  {name} health check FAILED (port {port})')
            results.append({'agent': name, 'port': port, 'health': 'unhealthy'})
            continue
        print(f'  {name} healthy; executing functional probes (warnings not fatal)...')
        p = discover_and_exercise(name, port)
        results.append(p)

    # Report
    summary = {'timestamp': time.time(), 'results': results}
    out = json.dumps(summary, indent=2)
    print('\n=== SUMMARY ===')
    print(out)
    if args.output:
        with open(args.output, 'w') as fh:
            fh.write(out)

    # Consider service unhealthy only if /health did not return 200.
    unhealthy = [r for r in results if r.get('health') != 200]
    if unhealthy:
        print('\nSome services are unhealthy. See summary for details.')
        sys.exit(2)

    # If we reach here, all /health endpoints returned 200. Probe failures are warnings.
    print('\nAll checked services responded to /health. Functional probe failures are reported as warnings in the summary.')
    sys.exit(0)

if __name__ == '__main__':
    main()
