import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration centralisée pour le système RAG ISO 9001"""
    
    # Chemins
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "chroma_db"
    COMPANY_DOCS_DIR = BASE_DIR / "Doc"
    CONFORMITY_REPORTS_DIR = BASE_DIR / "reports"
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # Modèle Groq
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE = 0.1
    GROQ_MAX_TOKENS = 2048
    
    # Embeddings
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_DIMENSION = 768
    
    # ChromaDB
    COLLECTION_NAME = "iso_9001_knowledge"
    DISTANCE_METRIC = "cosine"
    
    # Chunking
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    
    # Retrieval
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Documents sources
    SOURCE_DOCS = {
        "ISO_9001_2015": "ISO_9001_2015.pdf",
        "ISO_9000_2015": "ISO_9000_2015.pdf"
    }
    
    @classmethod
    def create_directories(cls):
        """Créer les répertoires nécessaires"""
        for directory in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, 
                         cls.VECTORSTORE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Valider la configuration"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY n'est pas définie dans .env")
        
        cls.create_directories()
        return True