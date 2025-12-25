from typing import List, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Gestionnaire de la base de données vectorielle ChromaDB"""
    
    def __init__(self, 
                 persist_directory: Path,
                 collection_name: str,
                 embedding_model: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Initialiser les embeddings
        logger.info(f"Chargement du modèle d'embeddings: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = None
        self.client = None
    
    def initialize_vectorstore(self) -> Chroma:
        """Initialiser ou charger le vectorstore"""
        try:
            # Configuration de ChromaDB
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Créer ou charger le vectorstore
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            logger.info(f"✓ VectorStore initialisé: {self.collection_name}")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du vectorstore: {e}")
            raise
    
    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """Ajouter des documents au vectorstore par batch"""
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        logger.info(f"Ajout de {len(documents)} documents à ChromaDB...")
        
        # Traiter par batch pour éviter les problèmes de mémoire
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                self.vectorstore.add_documents(batch)
                logger.info(f"✓ Batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ajouté")
            except Exception as e:
                logger.error(f"Erreur lors de l'ajout du batch {i//batch_size + 1}: {e}")
        
        logger.info("✓ Tous les documents ont été ajoutés")
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filter_dict: Optional[dict] = None) -> List[Document]:
        """Recherche par similarité"""
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            return results
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            return []
    
    def similarity_search_with_score(self, 
                                     query: str, 
                                     k: int = 5,
                                     filter_dict: Optional[dict] = None) -> List[tuple]:
        """Recherche par similarité avec scores"""
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            return results
        except Exception as e:
            logger.error(f"Erreur lors de la recherche avec scores: {e}")
            return []
    
    def get_collection_stats(self) -> dict:
        """Obtenir les statistiques de la collection"""
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                'collection_name': self.collection_name,
                'total_documents': count,
                'embedding_model': self.embedding_model_name,
                'persist_directory': str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats: {e}")
            return {}
    
    def delete_collection(self):
        """Supprimer la collection (pour réinitialisation)"""
        try:
            if self.client:
                self.client.delete_collection(self.collection_name)
                logger.info(f"✓ Collection {self.collection_name} supprimée")
        except Exception as e:
            logger.warning(f"Impossible de supprimer la collection: {e}")
    
    def reset_vectorstore(self):
        """Réinitialiser complètement le vectorstore"""
        logger.warning("Réinitialisation du vectorstore...")
        self.delete_collection()
        self.vectorstore = None
        self.initialize_vectorstore()
        logger.info("✓ VectorStore réinitialisé")
    
    def check_if_empty(self) -> bool:
        """Vérifier si le vectorstore est vide"""
        stats = self.get_collection_stats()
        return stats.get('total_documents', 0) == 0
    
    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """Créer un retriever à partir du vectorstore"""
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        default_kwargs = {'k': 5}
        if search_kwargs:
            default_kwargs.update(search_kwargs)
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=default_kwargs
        )