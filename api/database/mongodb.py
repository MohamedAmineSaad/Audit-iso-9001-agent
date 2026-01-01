from motor.motor_asyncio import AsyncIOMotorClient
import os

# MongoDB URL (should be in .env)
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("MONGO_DB_NAME", "audit_db")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DATABASE_NAME]

def get_mongo_db():
    return db
