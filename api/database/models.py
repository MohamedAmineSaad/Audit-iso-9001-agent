from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from .postgres import Base

class AuditSession(Base):
    """
    PostgreSQL model to store structured audit session metadata.
    """
    __tablename__ = "audit_sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    document_name = Column(String, nullable=False)
    overall_score = Column(Float, default=0.0)
    status = Column(String, default="completed")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Optional: Store a summary of findings in PG for quick access
    summary = Column(String, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "document_name": self.document_name,
            "overall_score": self.overall_score,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "summary": self.summary
        }
