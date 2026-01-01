import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = REPO_ROOT / "main.py"


def run_cli(args: List[str], stdin_text: Optional[str] = None) -> Dict[str, Any]:
    cmd = [sys.executable, str(MAIN_PY), *args]

    env = dict(**__import__("os").environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["RICH_FORCE_TERMINAL"] = "0"   # avoid rich trying to be fancy in subprocess

    # Optional: silence chroma telemetry noise
    env.setdefault("ANONYMIZED_TELEMETRY", "False")
    env.setdefault("CHROMA_TELEMETRY", "False")
    env.setdefault("POSTHOG_DISABLED", "1")

    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    return {
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "exit_code": int(completed.returncode),
    }


# ----------------------------
# CHAT (unchanged)
# ----------------------------
def run_chat_one_turn(question: str) -> Dict[str, Any]:
    res = run_cli(["chat"], stdin_text=f"{question}\nexit\n")
    out = res.get("stdout", "")

    # 1) Extract answer from the Rich box
    # It appears between "📋 Réponse" and the bottom border line.
    answer = ""
    m = re.search(r"📋 Réponse[\s\S]*?\n(.*?)\n└", out)
    if m:
        block = m.group(1)

        # remove box borders "│ ... │"
        lines = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("│") and line.endswith("│"):
                line = line[1:-1].strip()
            elif line.startswith("│"):
                line = line[1:].strip()
            lines.append(line)

        # join and clean spacing
        answer = "\n".join([ln for ln in lines if ln != ""]).strip()

    # 2) Extract sources (basic parsing)
    sources = []
    if "📚 Sources:" in out:
        after = out.split("📚 Sources:", 1)[1]
        # each source starts with: [1] ...
        parts = re.split(r"\n\s*\[(\d+)\]\s+", after)
        # parts looks like: ["", "1", "ISO_...text", "2", "ISO_...text", ...]
        for i in range(1, len(parts), 2):
            idx = parts[i]
            body = parts[i + 1].strip()
            title_line = body.splitlines()[0].strip() if body else ""
            # keep a short snippet
            snippet = "\n".join(body.splitlines()[:6]).strip()
            sources.append({
                "id": int(idx),
                "title": title_line,
                "snippet": snippet,
            })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "raw_output": out,
        "raw_error": res.get("stderr", ""),
        "exit_code": res.get("exit_code", 1),
    }


# ----------------------------
# STATS (unchanged)
# ----------------------------
def run_stats() -> Dict[str, Any]:
    res = run_cli(["stats"])
    out = res.get("stdout", "")

    def find_value(label: str) -> str | None:
        # Matches: │ Label │ Value │
        m = re.search(rf"│\s*{re.escape(label)}\s*│\s*(.*?)\s*│", out)
        return m.group(1).strip() if m else None

    collection = find_value("Collection")
    docs = find_value("Documents indexés")
    embedding_model = find_value("Modèle d'embeddings")
    llm_model = find_value("Modèle LLM")
    directory = find_value("Répertoire")

    # docs -> int if possible
    documents_indexed = None
    if docs:
        m = re.search(r"\d+", docs)
        documents_indexed = int(m.group(0)) if m else None

    return {
        "collection": collection,
        "documents_indexed": documents_indexed,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "directory": directory,
        "raw_output": out,
        "raw_error": res.get("stderr", ""),
        "exit_code": res.get("exit_code", 1),
    }


# ----------------------------
# SEARCH (unchanged)
# ----------------------------
def run_search(query: str, k: int = 5) -> Dict[str, Any]:
    res = run_cli(["search", "--query", query, "--k", str(k)])
    out = res.get("stdout", "")

    hits = []
    # Split by numbered results: "1. ...", "2. ..."
    chunks = re.split(r"\n(?=\d+\.\s)", out)

    for chunk in chunks:
        m = re.match(r"(\d+)\.\s+(.*?)\s+\(Similarité:\s*([0-9.]+)%\)", chunk.strip())
        if not m:
            continue

        rank = int(m.group(1))
        doc = m.group(2).strip()
        similarity = float(m.group(3))

        # Page + section line
        page = None
        section = None
        m2 = re.search(r"Page\s+(\d+)\s+\|\s+Section\s+([^\n]+)", chunk)
        if m2:
            page = int(m2.group(1))
            section = m2.group(2).strip()

        # snippet = rest after the 3 header lines (best-effort)
        lines = [ln.rstrip() for ln in chunk.splitlines()]
        snippet = "\n".join(lines[3:]).strip() if len(lines) > 3 else ""

        hits.append({
            "rank": rank,
            "document": doc,
            "similarity_percent": similarity,
            "page": page,
            "section": section,
            "snippet": snippet,
        })

        if len(hits) >= k:
            break

    return {
        "query": query,
        "k": k,
        "hits": hits,
        "raw_output": out,
        "raw_error": res.get("stderr", ""),
        "exit_code": res.get("exit_code", 1),
    }


# ----------------------------
# NEW: ANALYZE parsing (added)
# ----------------------------
def _parse_analyze_actions(out: str) -> List[Dict[str, Any]]:
    """
    Extracts the '🎯 Actions prioritaires' bullets:
      • [Clause X.Y] text
    """
    actions: List[Dict[str, Any]] = []
    if "🎯 Actions prioritaires:" not in out:
        return actions

    after = out.split("🎯 Actions prioritaires:", 1)[1]
    for line in after.splitlines():
        line = line.strip()
        if not line.startswith("•"):
            # stop if we hit another big section header
            # (optional, but helps avoid grabbing unrelated content)
            if line.startswith(("✅", "📋", "🔍", "📊")):
                break
            continue

        text = line.lstrip("•").strip()
        m = re.match(r"\[Clause\s+([0-9.]+)\]\s*(.+)", text)
        if m:
            actions.append({"clause": m.group(1), "text": m.group(2).strip()})
        else:
            actions.append({"clause": None, "text": text})
    return actions


def _parse_analyze_summary(out: str) -> Dict[str, Any]:
    """
    Extracts summary table:
      Score de conformité global
      Total de clauses analysées
      Clauses conformes / partiellement conformes / non conformes
    """
    def find_percent(label: str) -> Optional[float]:
        m = re.search(rf"{re.escape(label)}\s*\│\s*([0-9]+(?:\.[0-9]+)?)\s*%", out)
        return float(m.group(1)) if m else None

    def find_int(label: str) -> Optional[int]:
        m = re.search(rf"{re.escape(label)}\s*\│\s*([0-9]+)", out)
        return int(m.group(1)) if m else None

    return {
        "global_score_percent": find_percent("Score de conformité global"),
        "clauses_analyzed": find_int("Total de clauses analysées"),
        "clauses_conformes": find_int("Clauses conformes"),
        "clauses_partiellement_conformes": find_int("Clauses partiellement conformes"),
        "clauses_non_conformes": find_int("Clauses non conformes"),
    }


def run_analyze(document_path: str) -> Dict[str, Any]:
    """
    Keeps old behavior (runs CLI) + adds structured fields:
      - document
      - summary
      - actions
    """
    res = run_cli(["analyze", "--document", document_path])
    out = res.get("stdout", "")

    summary = _parse_analyze_summary(out)
    actions = _parse_analyze_actions(out)

    return {
        "document": document_path,
        "report": None,     # placeholder: you can add full structured report later
        "summary": summary,
        "actions": actions,
        "raw_output": out,
        "raw_error": res.get("stderr", ""),
        "exit_code": res.get("exit_code", 1),
    }


def run_analyze_agentic(document_path: str) -> Dict[str, Any]:
    return run_cli(["analyze-agentic", "--document", document_path])


# ----------------------------
# NEW: INDEX non-blocking (added)
# ----------------------------
def run_index() -> Dict[str, Any]:
    """
    If CLI asks: 'reindex o/n', we auto-answer 'n' to avoid Swagger hanging.
    Later we can add a real flag to force reindex.
    """
    return run_cli(["index"], stdin_text="n\n")
