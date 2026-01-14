# init_db.py
import asyncio
from api.database.postgres import init_postgres, engine, Base
from api.database.models import AuditSession  # Make sure your model is imported

async def main():
    print("Creating PostgreSQL tables...")
    await init_postgres()  # This reads your Base.metadata and creates tables
    print("PostgreSQL tables created successfully.")

if __name__ == "__main__":
    asyncio.run(main())