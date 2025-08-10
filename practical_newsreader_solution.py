"""Deprecated duplicate module.

The canonical implementation lives at:
  agents.newsreader.main_options.practical_newsreader_solution

This thin wrapper exists to avoid breaking older imports while keeping the
root clean. Prefer importing from the agents path going forward.
"""

try:
    # Re-export the canonical symbols for backward compatibility
    from agents.newsreader.main_options.practical_newsreader_solution import *  # noqa: F401,F403
except Exception as e:  # pragma: no cover - guidance-only path
    raise ImportError(
        "Import from 'agents.newsreader.main_options.practical_newsreader_solution' instead. "
        f"Original error: {e}"
    )
