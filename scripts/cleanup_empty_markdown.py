#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "markdown_docs"

def is_empty_markdown(p: Path) -> bool:
    try:
        if not p.is_file():
            return False
        if p.stat().st_size == 0:
            return True
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content.strip() == ""
    except Exception:
        return False


def main() -> int:
    if not DOCS_DIR.exists():
        print(f"No markdown_docs directory at {DOCS_DIR}")
        return 0

    removed = []
    kept = []
    for path in DOCS_DIR.rglob('*.md'):
        if is_empty_markdown(path):
            try:
                path.unlink()
                removed.append(str(path.relative_to(ROOT)))
            except Exception as e:
                print(f"Failed to remove {path}: {e}")
        else:
            kept.append(str(path.relative_to(ROOT)))

    print(f"Removed {len(removed)} empty Markdown file(s).")
    for r in sorted(removed):
        print(f"REMOVED: {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
