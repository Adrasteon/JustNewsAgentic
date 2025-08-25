"""Utility to register a discoverable /info endpoint on FastAPI agents.

The endpoint returns agent metadata and a list of advertised routes (method + path).
If the caller passes an explicit list of probes, those are returned instead.
"""
from __future__ import annotations

from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for static type checking
    from fastapi import FastAPI  # type: ignore


def _route_list_from_app(app: "FastAPI") -> List[Dict[str, str]]:
    routes = []
    for r in app.routes:
        try:
            path = getattr(r, 'path', None) or getattr(r, 'path_regex', None)
            methods = getattr(r, 'methods', None) or getattr(r, 'methods', None)
            if not path or path.startswith('/openapi') or path.startswith('/docs') or path.startswith('/redoc'):
                continue
            if path == '/shutdown':
                continue
            if methods:
                for m in sorted(methods):
                    # FastAPI includes 'HEAD' and 'OPTIONS' sometimes; prefer GET/POST
                    routes.append({'method': m, 'path': path})
        except Exception:
            continue
    # Deduplicate while preserving order
    seen = set()
    out = []
    for r in routes:
        key = (r['method'], r['path'])
        if key in seen:
            continue
        seen.add(key)
        out.append({'method': r['method'], 'path': r['path']})
    return out


def register_info_endpoint(app: "FastAPI", agent_name: str, probes: Optional[List[Dict[str, str]]] = None) -> None:
    """Registers GET /info on the provided FastAPI `app`.

    The returned JSON has the shape:
      {"agent": agent_name, "probes": [{"method":"GET", "path":"/health"}, ...]}

    If `probes` is provided, it's returned verbatim. Otherwise the function inspects
    the app's routes to build a best-effort list.
    """

    if probes is None:
        try:
            probes = _route_list_from_app(app)
        except Exception:
            probes = [{'method': 'GET', 'path': '/health'}]

    @app.get('/info')
    def _info():
        return {'agent': agent_name, 'probes': probes}

    # Lightweight audit endpoint to record shutdown requests (IP and User-Agent).
    # This does NOT perform the actual shutdown; it's an audit hook so the orchestrator
    # can POST here to record who asked for shutdown. Agents may still implement their
    # own /shutdown endpoint which performs graceful shutdown.
    from fastapi import Request

    @app.post('/shutdown_audit')
    async def _shutdown_audit(request: Request):
        """Record an incoming shutdown audit. If the caller provided a JSON body,
        include that body under the 'origin' key in the persistent trace entry.
        """
        client = 'unknown'
        ua = 'unknown'
        origin_payload = None
        try:
            client = request.client.host if request.client else 'unknown'
            ua = request.headers.get('user-agent', 'unknown')
            try:
                origin_payload = await request.json()
            except Exception:
                origin_payload = None
        except Exception:
            client = 'unknown'
            ua = 'unknown'
            origin_payload = None
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info('shutdown_audit received', extra={'agent': agent_name, 'from': client, 'user-agent': ua, 'origin': origin_payload})
        except Exception:
            pass
        # Also write a persistent shutdown trace so CI runs preserve audit evidence even if
        # stdout/stderr log files are rotated or lost. Include any parsed origin payload.
        try:
            import json, time, os
            traces_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(traces_dir, exist_ok=True)
            trace_path = os.path.join(traces_dir, 'shutdown_traces.log')
            entry = {'ts': time.time(), 'agent': agent_name, 'event': 'shutdown_audit', 'from': client, 'user-agent': ua}
            if origin_payload is not None:
                entry['origin'] = origin_payload
            with open(trace_path, 'a') as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception:
            # best-effort; don't raise from audit
            pass
        # 204 No Content - auditing only
        from fastapi.responses import Response
        return Response(status_code=204)


    def _write_shutdown_trace(agent_name: str, reason: str, extra: Optional[Dict[str, str]] = None) -> None:
        """Append a single JSON line to logs/shutdown_traces.log.

        This is a best-effort helper used both by HTTP shutdown auditing and by
        on_shutdown / signal handlers inside agents so we have an indelible record
        of process exits.
        """
        try:
            import json, time, os
            traces_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(traces_dir, exist_ok=True)
            trace_path = os.path.join(traces_dir, 'shutdown_traces.log')
            entry = {'ts': time.time(), 'agent': agent_name, 'event': 'shutdown', 'reason': reason}
            if extra:
                entry.update(extra)
            with open(trace_path, 'a') as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception:
            return


    def _write_agent_exit_file(agent_name: str, reason: str, logs_to_tail: int = 50) -> None:
        """Write a small JSON file `logs/{agent}_exit_{ts}.json` containing pid, reason and
        the last `logs_to_tail` lines of the agent logfile for quick forensic analysis."""
        try:
            import json, time, os, subprocess, logging
            traces_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(traces_dir, exist_ok=True)
            ts = int(time.time())
            out_path = os.path.join(traces_dir, f"{agent_name}_exit_{ts}.json")
            logfile = os.path.join(os.getcwd(), 'logs', f"{agent_name}.log")
            tail_lines = []
            if os.path.exists(logfile):
                try:
                    p = subprocess.Popen(['tail', '-n', str(logs_to_tail), logfile], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    out, _ = p.communicate(timeout=2)
                    tail_lines = out.decode('utf-8', errors='replace').splitlines()
                except Exception:
                    try:
                        with open(logfile, 'rb') as fh:
                            data = fh.read().splitlines()
                            tail_lines = [line.decode('utf-8', errors='replace') for line in data[-logs_to_tail:]]
                    except Exception:
                        tail_lines = []
            entry = {'ts': ts, 'agent': agent_name, 'reason': reason, 'pid': os.getpid(), 'tail': tail_lines}
            with open(out_path, 'w') as fh:
                json.dump(entry, fh)
            try:
                logger = logging.getLogger(__name__)
                logger.info(f"wrote exit file {out_path}", extra={"agent": agent_name, "path": out_path})
            except Exception:
                pass
        except Exception:
            try:
                import logging
                logging.getLogger(__name__).exception("failed to write agent exit file", exc_info=True)
            except Exception:
                pass
            return


    def register_shutdown_trace_handlers(app: "FastAPI", agent_name: str) -> None:
        """Register a FastAPI on_shutdown handler and a SIGTERM handler that writes a
        persistent shutdown trace. Call this from agent main modules after configuring
        logging.
        """
        try:
            import logging
            logging.getLogger(__name__).info(f"register_shutdown_trace_handlers called for {agent_name}")
            # FastAPI on_shutdown event
            @app.on_event('shutdown')
            def _on_shutdown():
                _write_shutdown_trace(agent_name, 'fastapi_on_shutdown')
                _write_agent_exit_file(agent_name, 'fastapi_on_shutdown')

            # Also handle SIGTERM so abrupt terminations are recorded
            import signal

            def _sigterm_handler(signum, frame):
                _write_shutdown_trace(agent_name, f'signal_{signum}')
                _write_agent_exit_file(agent_name, f'signal_{signum}')
                # Re-raise default handler after trace so process exits normally
                try:
                    signal.default_int_handler(signum, frame)
                except Exception:
                    # import here to avoid top-level import side-effects
                    import os
                    os._exit(0)

            try:
                signal.signal(signal.SIGTERM, _sigterm_handler)
            except Exception:
                # Some runtimes (uvicorn workers) may not permit signal registration from
                # certain threads — best-effort only.
                pass
        except Exception:
            # If anything fails, don't prevent the agent from starting
            return
