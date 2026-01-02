from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB URL (should be in .env)
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("MONGO_DB_NAME", "audit_db")

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

db_instance = MongoDB()

async def connect_to_mongo():
    """Initialize MongoDB connection."""
    db_instance.client = AsyncIOMotorClient(MONGO_URL)
    db_instance.db = db_instance.client[DATABASE_NAME]
    print(f"Connected to MongoDB: {DATABASE_NAME}")

async def close_mongo_connection():
    """Close MongoDB connection."""
    if db_instance.client:
        db_instance.client.close()
        print("Closed MongoDB connection")

def get_mongo_db():
    """Dependency for getting the MongoDB database instance."""
    if db_instance.db is None:
        # Fallback for direct calls if not initialized via app lifecycle
        client = AsyncIOMotorClient(MONGO_URL)
        return client[DATABASE_NAME]
    return db_instance.db
