from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class ChatIn(BaseModel):
    question: str

class AnalyzeIn(BaseModel):
    document: str

class AnalyzeAgenticIn(BaseModel):
    document: str

class IndexIn(BaseModel):
    force: bool = False

class Hit(BaseModel):
    rank: int
    document: str
    similarity_percent: float
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: str

class SearchOut(BaseModel):
    query: str
    k: int
    hits: List[Hit]

class StatsOut(BaseModel):
    collection: Optional[str] = None
    documents_indexed: Optional[int] = None
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    directory: Optional[str] = None

class NormalizedResponse(BaseModel):
    kind: str
    mode: str
    data: Dict[str, Any]
    raw_output: str
    raw_error: str
    exit_code: int
