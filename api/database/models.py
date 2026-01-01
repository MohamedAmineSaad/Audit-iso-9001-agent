from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from .postgres import Base

class AuditSession(Base):
    __tablename__ = "audit_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    document_name = Column(String, nullable=False)
    overall_score = Column(Float)
    status = Column(String, default="completed")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<AuditSession(id={self.id}, document='{self.document_name}', score={self.overall_score})>"
