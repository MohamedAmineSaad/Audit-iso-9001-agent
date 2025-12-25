from typing import List, Dict, Optional
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISO9001RAGSystem:
    """Système RAG spécialisé pour ISO 9001/9000"""
    
    def __init__(self, 
                 groq_api_key: str,
                 vectorstore_manager,
                 model_name: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.1):
        
        self.vectorstore_manager = vectorstore_manager
        
        # Initialiser le LLM Groq
        logger.info(f"Initialisation du modèle Groq: {model_name}")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=2048
        )
        
        # Template de prompt spécialisé ISO
        self.prompt_template = self._create_prompt_template()
        
        # Chaîne QA
        self.qa_chain = None
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Créer un template de prompt spécialisé pour ISO 9001"""
        template = """Tu es un expert en systèmes de management de la qualité ISO 9001:2015 et ISO 9000:2015. 
Ton rôle est d'aider les utilisateurs à comprendre et appliquer ces normes.

CONTEXTE PERTINENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Réponds en français de manière claire et structurée
2. Base ta réponse UNIQUEMENT sur le contexte fourni des normes ISO 9001 et ISO 9000
3. Cite les sections pertinentes (ex: "Selon la section 4.1 de l'ISO 9001...")
4. Si la réponse n'est pas dans le contexte, dis-le clairement
5. Utilise une terminologie précise conforme aux normes ISO
6. Structure ta réponse avec des paragraphes et listes si nécessaire
7. Ajoute des exemples pratiques quand c'est pertinent

RÉPONSE:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def initialize_qa_chain(self, search_kwargs: Optional[dict] = None):
        """Initialiser la chaîne de questions-réponses"""
        if search_kwargs is None:
            search_kwargs = {'k': 5}
        
        retriever = self.vectorstore_manager.as_retriever(
            search_kwargs=search_kwargs
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
        
        logger.info("✓ Chaîne QA initialisée")
    
    def ask(self, question: str, return_sources: bool = True) -> Dict:
        """Poser une question au système RAG"""
        if not self.qa_chain:
            self.initialize_qa_chain()
        
        try:
            logger.info(f"Question: {question}")
            
            # Obtenir la réponse
            result = self.qa_chain.invoke({"query": question})
            
            response = {
                'question': question,
                'answer': result['result'],
                'sources': []
            }
            
            # Ajouter les sources si demandé
            if return_sources and 'source_documents' in result:
                response['sources'] = self._format_sources(
                    result['source_documents']
                )
            
            logger.info("✓ Réponse générée")
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse: {e}")
            return {
                'question': question,
                'answer': f"Erreur: {str(e)}",
                'sources': []
            }
    
    def _format_sources(self, documents: List[Document]) -> List[Dict]:
        """Formatter les documents sources pour l'affichage"""
        sources = []
        for i, doc in enumerate(documents, 1):
            source_info = {
                'rank': i,
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'section': doc.metadata.get('section_number', 'N/A'),
                'section_title': doc.metadata.get('section_title', 'N/A'),
                'content_preview': doc.page_content[:200] + "..." 
                                  if len(doc.page_content) > 200 
                                  else doc.page_content
            }
            sources.append(source_info)
        return sources
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Rechercher des documents pertinents sans génération"""
        results = self.vectorstore_manager.similarity_search_with_score(
            query=query,
            k=k
        )
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'similarity_score': float(1 - score),  # Convertir distance en similarité
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'section': doc.metadata.get('section_number', 'N/A'),
                'section_title': doc.metadata.get('section_title', 'N/A'),  # ← AJOUTÉ
                'content': doc.page_content
            })
        
        return formatted_results
    
    def ask_with_filter(self, 
                       question: str, 
                       source_filter: Optional[str] = None,
                       section_filter: Optional[str] = None) -> Dict:
        """Poser une question avec filtres sur les sources"""
        # Construire le filtre
        filter_dict = {}
        if source_filter:
            filter_dict['source'] = source_filter
        if section_filter:
            filter_dict['section_number'] = section_filter
        
        # Réinitialiser la chaîne avec le filtre
        search_kwargs = {'k': 5}
        if filter_dict:
            search_kwargs['filter'] = filter_dict
        
        self.initialize_qa_chain(search_kwargs)
        
        return self.ask(question)
    
    def get_section_content(self, section_number: str) -> List[str]:
        """Récupérer le contenu d'une section spécifique"""
        results = self.vectorstore_manager.similarity_search(
            query=f"section {section_number}",
            k=10,
            filter_dict={'section_number': section_number}
        )
        
        return [doc.page_content for doc in results]
    
    def batch_questions(self, questions: List[str]) -> List[Dict]:
        """Traiter plusieurs questions en batch"""
        results = []
        for question in questions:
            result = self.ask(question, return_sources=False)
            results.append(result)
        return results
    
    def explain_section(self, section_number: str) -> Dict:
        """Expliquer une section spécifique de la norme"""
        question = f"Explique en détail la section {section_number} de la norme ISO 9001. Quelles sont les exigences principales et comment les mettre en œuvre ?"
        return self.ask_with_filter(question, section_filter=section_number)