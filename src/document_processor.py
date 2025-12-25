from typing import List, Dict
import re
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISO9001DocumentProcessor:
    """Traitement spécialisé pour les documents ISO 9001/9000"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Triple saut de ligne
                "\n\n",    # Double saut de ligne
                "\n",      # Saut de ligne simple
                ". ",      # Fin de phrase
                ", ",      # Virgule
                " ",       # Espace
                ""         # Caractère
            ],
            keep_separator=True
        )
    
    def load_pdf(self, pdf_path: Path) -> List[Document]:
        """Charger un document PDF"""
        try:
            logger.info(f"Chargement du PDF: {pdf_path.name}")
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            logger.info(f"✓ {len(documents)} pages chargées")
            return documents
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {pdf_path}: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Nettoyer le texte extrait du PDF"""
        # Supprimer les caractères spéciaux excessifs
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Supprimer les espaces multiples
        text = re.sub(r' {2,}', ' ', text)
        
        # Supprimer les lignes vides multiples
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Garder les numéros de section ISO
        text = re.sub(r'(\d+\.\d+(?:\.\d+)?)\s+', r'\1 ', text)
        
        return text.strip()
    
    def extract_section_info(self, text: str) -> Dict[str, str]:
        """Extraire les informations de section ISO"""
        metadata = {}
        
        # Identifier le numéro de section (ex: 4.1, 5.2.1)
        section_match = re.search(r'(\d+(?:\.\d+){0,2})\s+([^\n]+)', text)
        if section_match:
            metadata['section_number'] = section_match.group(1)
            metadata['section_title'] = section_match.group(2).strip()
        
        # Identifier les mots-clés importants
        keywords = []
        for keyword in ['exigence', 'doit', 'il convient', 'organisme', 
                       'qualité', 'management', 'processus', 'client']:
            if keyword.lower() in text.lower():
                keywords.append(keyword)
        metadata['keywords'] = ', '.join(keywords)
        
        return metadata
    
    def process_documents(self, documents: List[Document], 
                         source_name: str) -> List[Document]:
        """Traiter et découper les documents"""
        logger.info(f"Traitement de {len(documents)} documents de {source_name}")
        
        # Nettoyer et enrichir les documents
        cleaned_docs = []
        for doc in tqdm(documents, desc="Nettoyage"):
            cleaned_text = self.clean_text(doc.page_content)
            
            # Extraire les métadonnées de section
            section_info = self.extract_section_info(cleaned_text)
            
            # Enrichir les métadonnées
            doc.metadata.update({
                'source': source_name,
                'cleaned': True,
                **section_info
            })
            doc.page_content = cleaned_text
            cleaned_docs.append(doc)
        
        # Découper en chunks
        logger.info("Découpage en chunks...")
        chunks = self.text_splitter.split_documents(cleaned_docs)
        
        # Ajouter l'index de chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        
        logger.info(f"✓ {len(chunks)} chunks créés")
        return chunks
    
    def process_iso_documents(self, pdf_paths: Dict[str, Path]) -> List[Document]:
        """Traiter tous les documents ISO"""
        all_chunks = []
        
        for source_name, pdf_path in pdf_paths.items():
            if not pdf_path.exists():
                logger.warning(f"Fichier non trouvé: {pdf_path}")
                continue
            
            # Charger le PDF
            documents = self.load_pdf(pdf_path)
            
            if not documents:
                continue
            
            # Traiter les documents
            chunks = self.process_documents(documents, source_name)
            all_chunks.extend(chunks)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Total de chunks créés: {len(all_chunks)}")
        logger.info(f"{'='*60}\n")
        
        return all_chunks
    
    def get_statistics(self, chunks: List[Document]) -> Dict:
        """Obtenir des statistiques sur les chunks"""
        stats = {
            'total_chunks': len(chunks),
            'sources': {},
            'avg_length': 0,
            'sections': set()
        }
        
        total_length = 0
        for chunk in chunks:
            source = chunk.metadata.get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
            total_length += len(chunk.page_content)
            
            if 'section_number' in chunk.metadata:
                stats['sections'].add(chunk.metadata['section_number'])
        
        stats['avg_length'] = total_length / len(chunks) if chunks else 0
        stats['unique_sections'] = len(stats['sections'])
        
        return stats