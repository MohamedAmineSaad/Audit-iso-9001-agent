from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database URL (should be in .env)
# Example: postgresql+asyncpg://user:password@localhost/audit_db
DATABASE_URL = os.getenv("POSTGRES_URL")

if not DATABASE_URL:
    # Fallback for development, but should be set in .env
    DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost/audit_db"

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def get_db():
    """Dependency for getting an async PostgreSQL session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_postgres():
    """Utility to create tables if they don't exist (for simple setups)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
