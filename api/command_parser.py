from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Parsed:
    mode: str
    args: Dict[str, Any]


def parse_message(text: str) -> Parsed:
    """
    If text starts with '/', treat as command.
    Otherwise: chat.
    Commands:
      /stats
      /index [--force]
      /search "query" [k]
      /analyze "Doc/file.docx"
      /analyze-agentic "Doc/file.docx"
      /chat <question>   (optional explicit)
    """
    t = text.strip()
    if not t.startswith("/"):
        return Parsed("chat", {"question": t})

    parts = t[1:].split()
    if not parts:
        return Parsed("chat", {"question": ""})

    cmd = parts[0].lower()

    if cmd == "chat":
        q = t[len("/chat"):].strip()
        return Parsed("chat", {"question": q})

    if cmd == "stats":
        return Parsed("stats", {})

    if cmd == "index":
        return Parsed("index", {"force": "--force" in parts[1:]})

    if cmd == "search":
        rest = t[len("/search"):].strip()
        k = 5
        toks = rest.split()
        if toks:
            try:
                k = int(toks[-1])
                rest = " ".join(toks[:-1]).strip()
            except ValueError:
                pass
        query = rest.strip().strip('"').strip("'")
        return Parsed("search", {"query": query, "k": k})

    if cmd == "analyze":
        doc = t[len("/analyze"):].strip().strip('"').strip("'")
        return Parsed("analyze", {"document": doc})

    if cmd in ("analyze-agentic", "agentic"):
        doc = t[len("/analyze-agentic"):].strip().strip('"').strip("'")
        return Parsed("analyze-agentic", {"document": doc})

    # unknown command -> treat as chat (safe fallback)
    return Parsed("chat", {"question": t})
