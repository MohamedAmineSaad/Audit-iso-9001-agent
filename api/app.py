from fastapi import FastAPI, Query, Depends
from typing import Any, Dict
import shlex
from sqlalchemy.ext.asyncio import AsyncSession
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.runner import (
    run_chat_one_turn,
    run_stats,
    run_search,
    run_analyze,
    run_analyze_agentic,
    run_index,
)

from api.schemas import (
    ChatIn, AnalyzeIn, AnalyzeAgenticIn, IndexIn,
    NormalizedResponse
)

from api.database.postgres import get_db
from api.database.mongodb import get_mongo_db
from api.database.models import AuditSession

app = FastAPI(title="Audit ISO 9001 API")


# -----------------------
# Health
# -----------------------
@app.get("/health")
def health():
    return {"ok": True}


def _resp(kind: str, mode: str, data: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, Any]:
    raw_output = res.get("raw_output", res.get("stdout", ""))
    raw_error = res.get("raw_error", res.get("stderr", ""))
    exit_code = res.get("exit_code", 1)

    return {
        "kind": kind,
        "mode": mode,
        "data": data,
        "raw_output": raw_output,
        "raw_error": raw_error,
        "exit_code": exit_code,
    }


# -----------------------
# New clean endpoints
# -----------------------
@app.get("/v1/stats", response_model=NormalizedResponse)
def stats():
    res = run_stats()
    data = {
        "collection": res.get("collection"),
        "documents_indexed": res.get("documents_indexed"),
        "embedding_model": res.get("embedding_model"),
        "llm_model": res.get("llm_model"),
        "directory": res.get("directory"),
    }
    return _resp("command", "stats", data, res)


@app.get("/v1/search", response_model=NormalizedResponse)
def search(query: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=20)):
    res = run_search(query, k)
    data = {
        "query": res.get("query", query),
        "k": res.get("k", k),
        "hits": res.get("hits", []),
    }
    return _resp("command", "search", data, res)


@app.post("/v1/chat", response_model=NormalizedResponse)
def chat(payload: ChatIn):
    q = (payload.question or "").strip()
    res = run_chat_one_turn(q)
    data = {
        "question": res.get("question", q),
        "answer": res.get("answer", ""),
        "sources": res.get("sources", []),
    }
    return _resp("chat", "chat", data, res)


@app.post("/v1/analyze", response_model=NormalizedResponse)
async def analyze(
    payload: AnalyzeIn,
    pg_db: AsyncSession = Depends(get_db),
    mg_db: AsyncIOMotorDatabase = Depends(get_mongo_db)
):
    doc_path = payload.document
    res = run_analyze(doc_path)
    
    # Store in PostgreSQL
    new_session = AuditSession(
        user_id="anonymous",
        document_name=doc_path,
        overall_score=0.0 # Placeholder, update based on res
    )
    pg_db.add(new_session)
    await pg_db.commit()
    await pg_db.refresh(new_session)
    
    # Store in MongoDB
    detailed_report = {
        "session_id": new_session.id,
        "full_report": res,
        "metadata": {"document": doc_path}
    }
    await mg_db["reports"].insert_one(detailed_report)

    data = {
        "document": res.get("document", doc_path),
        "report": res.get("report"),
        "summary": res.get("summary"),
        "actions": res.get("actions", []),
        "session_id": new_session.id
    }
    return _resp("command", "analyze", data, res)


@app.post("/v1/analyze-agentic", response_model=NormalizedResponse)
async def analyze_agentic(
    payload: AnalyzeAgenticIn,
    pg_db: AsyncSession = Depends(get_db),
    mg_db: AsyncIOMotorDatabase = Depends(get_mongo_db)
):
    doc_path = payload.document
    res = run_analyze_agentic(doc_path)
    
    # Store in PostgreSQL
    new_session = AuditSession(
        user_id="anonymous",
        document_name=doc_path,
        overall_score=0.0
    )
    pg_db.add(new_session)
    await pg_db.commit()
    await pg_db.refresh(new_session)
    
    # Store in MongoDB
    detailed_report = {
        "session_id": new_session.id,
        "full_report": res,
        "metadata": {"document": doc_path, "mode": "agentic"}
    }
    await mg_db["reports"].insert_one(detailed_report)

    data = {
        "document": doc_path,
        "note": "Agentic analysis executed.",
        "session_id": new_session.id
    }
    return _resp("command", "analyze-agentic", data, res)


@app.post("/v1/index", response_model=NormalizedResponse)
def index(payload: IndexIn):
    res = run_index()
    data = {
        "note": "Index command executed.",
        "force": payload.force,
    }
    return _resp("command", "index", data, res)


# -----------------------
# Keep your old endpoint (/v1/message)
# -----------------------
from pydantic import BaseModel

class MessageIn(BaseModel):
    user_id: str
    session_id: str
    text: str

@app.post("/v1/message", response_model=NormalizedResponse)
async def message(
    payload: MessageIn,
    pg_db: AsyncSession = Depends(get_db),
    mg_db: AsyncIOMotorDatabase = Depends(get_mongo_db)
) -> Dict[str, Any]:
    text = (payload.text or "").strip()

    if text.startswith("/"):
        try:
            parts = shlex.split(text)
        except ValueError:
            parts = [text]

        command = (parts[0] or "").lower()

        if command == "/stats":
            return stats()

        if command == "/index":
            return index(IndexIn(force=False))

        if command == "/search" and len(parts) >= 2:
            query = parts[1]
            k = int(parts[2]) if len(parts) >= 3 and str(parts[2]).isdigit() else 5
            return search(query=query, k=k)

        if command == "/analyze" and len(parts) >= 2:
            return await analyze(AnalyzeIn(document=parts[1]), pg_db, mg_db)

        if command in ("/analyze-agentic", "/analyze_agentic") and len(parts) >= 2:
            return await analyze_agentic(AnalyzeAgenticIn(document=parts[1]), pg_db, mg_db)

        return _resp(
            "command",
            "unknown",
            {"message": f"Unknown command: {command}"},
            {"exit_code": 0, "stdout": "", "stderr": ""},
        )

    return chat(ChatIn(question=text))
