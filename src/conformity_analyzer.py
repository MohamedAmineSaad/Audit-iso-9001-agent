"""
Module d'analyse de conformité ISO 9001/9000
Contient les agents d'analyse : Mappeur, Vérificateur, Scoreur
"""
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from src.document_extractor import DocumentSection

logger = logging.getLogger(__name__)


class ConformityStatus(Enum):
    """Statuts de conformité possibles"""
    CONFORME = "conforme"
    PARTIELLEMENT_CONFORME = "partiellement_conforme"
    NON_CONFORME = "non_conforme"
    NON_APPLICABLE = "non_applicable"
    A_VERIFIER = "a_verifier"


@dataclass
class ISOClauseMapping:
    """Représente le mapping d'une section vers une clause ISO"""
    clause_number: str
    clause_title: str
    relevance_score: float  # 0-1
    justification: str
    iso_requirements: str  # Exigences extraites du RAG
    source_document: str  # ISO 9001 ou ISO 9000


@dataclass
class ConformityAnalysis:
    """Résultat d'analyse de conformité pour une clause"""
    clause_number: str
    clause_title: str
    status: ConformityStatus
    conformity_score: int  # 0-100
    conformity_elements: List[str] = field(default_factory=list)
    non_conformities: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)  # Citations du document


@dataclass
class SectionAnalysisResult:
    """Résultat complet de l'analyse d'une section"""
    section_title: str
    section_type: str
    mapped_clauses: List[ISOClauseMapping] = field(default_factory=list)
    conformity_analyses: List[ConformityAnalysis] = field(default_factory=list)
    overall_score: float = 0.0
    summary: str = ""


class ConformityVerificationAgent:
    """Agent 2 : Vérificateur de conformité"""
    
    def __init__(self, llm: ChatGroq, rag_system):
        """
        Initialise l'agent vérificateur
        
        Args:
            llm: Modèle de langage Groq
            rag_system: Système RAG pour récupérer les exigences ISO
        """
        self.llm = llm
        self.rag_system = rag_system
        self.verification_prompt = self._create_verification_prompt()
    
    def _create_verification_prompt(self) -> PromptTemplate:
        """Créer le prompt pour la vérification de conformité"""
        template = """Tu es un auditeur ISO 9001 expert. Ta mission est d'évaluer la conformité d'un document d'entreprise par rapport aux exigences ISO.

SECTION DU DOCUMENT À ÉVALUER:
Titre: {section_title}
Contenu: {section_content}

CLAUSE ISO CONCERNÉE:
Clause {clause_number}: {clause_title}

EXIGENCES ISO POUR CETTE CLAUSE:
{iso_requirements}

TÂCHE:
Analyse si le document répond aux exigences de cette clause ISO. Pour chaque exigence:
1. Identifie si elle est satisfaite, partiellement satisfaite, ou non satisfaite
2. Extrais les preuves concrètes du document (citations exactes)
3. Identifie les lacunes ou non-conformités
4. Propose des recommandations d'amélioration

CRITÈRES D'ÉVALUATION:
- Conforme (80-100%): Toutes les exigences majeures sont satisfaites
- Partiellement conforme (50-79%): Certaines exigences sont satisfaites, d'autres manquent
- Non conforme (0-49%): La plupart des exigences ne sont pas satisfaites
- Non applicable: La clause ne s'applique pas à cette section

RÉPONDS UNIQUEMENT EN FORMAT JSON (sans markdown, sans backticks):
{{
  "status": "conforme|partiellement_conforme|non_conforme|non_applicable",
  "conformity_score": 0-100,
  "conformity_elements": [
    "Élément conforme 1 avec preuve",
    "Élément conforme 2 avec preuve"
  ],
  "non_conformities": [
    "Non-conformité 1 détectée",
    "Non-conformité 2 détectée"
  ],
  "recommendations": [
    "Recommandation 1 pour améliorer",
    "Recommandation 2 pour améliorer"
  ],
  "evidence": [
    "Citation exacte 1 du document",
    "Citation exacte 2 du document"
  ]
}}"""
        
        return PromptTemplate(
            template=template,
            input_variables=["section_title", "section_content", "clause_number", 
                           "clause_title", "iso_requirements"]
        )
    
    def verify_conformity(self, 
                         section: DocumentSection, 
                         mapping: ISOClauseMapping) -> ConformityAnalysis:
        """
        Vérifier la conformité d'une section pour une clause ISO spécifique
        
        Args:
            section: Section du document à vérifier
            mapping: Mapping de la clause ISO
            
        Returns:
            Analyse de conformité
        """
        logger.info(f"Vérification conformité: {section.title} vs Clause {mapping.clause_number}")
        
        try:
            # Enrichir les exigences ISO si nécessaire
            iso_requirements = mapping.iso_requirements
            if not iso_requirements or len(iso_requirements) < 100:
                iso_requirements = self._fetch_detailed_requirements(mapping.clause_number)
            
            # Limiter la taille du contenu pour le prompt
            section_content = section.content[:2000] if len(section.content) > 2000 else section.content
            
            # Appeler le LLM pour vérification
            prompt = self.verification_prompt.format(
                section_title=section.title,
                section_content=section_content,
                clause_number=mapping.clause_number,
                clause_title=mapping.clause_title,
                iso_requirements=iso_requirements[:1500]  # Limiter aussi les exigences
            )
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Parser la réponse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # Créer l'objet ConformityAnalysis
            status_map = {
                "conforme": ConformityStatus.CONFORME,
                "partiellement_conforme": ConformityStatus.PARTIELLEMENT_CONFORME,
                "non_conforme": ConformityStatus.NON_CONFORME,
                "non_applicable": ConformityStatus.NON_APPLICABLE
            }
            
            analysis = ConformityAnalysis(
                clause_number=mapping.clause_number,
                clause_title=mapping.clause_title,
                status=status_map.get(result.get("status", "a_verifier"), ConformityStatus.A_VERIFIER),
                conformity_score=int(result.get("conformity_score", 0)),
                conformity_elements=result.get("conformity_elements", []),
                non_conformities=result.get("non_conformities", []),
                recommendations=result.get("recommendations", []),
                evidence=result.get("evidence", [])
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de conformité: {e}")
            # Retourner une analyse par défaut en cas d'erreur
            return ConformityAnalysis(
                clause_number=mapping.clause_number,
                clause_title=mapping.clause_title,
                status=ConformityStatus.A_VERIFIER,
                conformity_score=0,
                non_conformities=[f"Erreur lors de l'analyse: {str(e)}"]
            )
    
    def _fetch_detailed_requirements(self, clause_number: str) -> str:
        """Récupérer les exigences détaillées d'une clause via le RAG"""
        try:
            query = f"Quelles sont toutes les exigences détaillées de la clause {clause_number} de la norme ISO 9001:2015? Liste complète."
            
            # Utiliser la méthode ask du RAG pour obtenir une réponse structurée
            result = self.rag_system.ask(query, return_sources=True)
            
            # Combiner la réponse et les sources
            requirements = result['answer']
            
            # Ajouter le contenu des sources pour plus de détails
            if result.get('sources'):
                for source in result['sources'][:2]:  # Prendre les 2 premières sources
                    requirements += f"\n\n{source['content_preview']}"
            
            return requirements
            
        except Exception as e:
            logger.warning(f"Impossible de récupérer les exigences pour {clause_number}: {e}")
            return f"Clause {clause_number}: Exigences non disponibles"
    
    def verify_multiple_clauses(self, 
                                section: DocumentSection, 
                                mappings: List[ISOClauseMapping]) -> List[ConformityAnalysis]:
        """
        Vérifier la conformité pour plusieurs clauses
        
        Args:
            section: Section du document
            mappings: Liste des mappings de clauses
            
        Returns:
            Liste des analyses de conformité
        """
        analyses = []
        
        for mapping in mappings:
            # Ne vérifier que les clauses avec un score de pertinence suffisant
            if mapping.relevance_score >= 0.6:
                analysis = self.verify_conformity(section, mapping)
                analyses.append(analysis)
        
        return analyses


class ClauseMappingAgent:
    """Agent 1 : Mappeur de clauses ISO"""
    
    # Mapping des mots-clés vers les clauses ISO 9001 principales
    ISO_9001_CLAUSE_MAP = {
        "4.1": {
            "title": "Compréhension de l'organisation et de son contexte",
            "keywords": ["contexte", "organisation", "enjeux", "parties intéressées", "environnement"]
        },
        "4.2": {
            "title": "Compréhension des besoins et attentes des parties intéressées",
            "keywords": ["parties intéressées", "besoins", "attentes", "exigences", "clients"]
        },
        "4.3": {
            "title": "Détermination du domaine d'application du SMQ",
            "keywords": ["domaine", "application", "périmètre", "limites", "smq"]
        },
        "4.4": {
            "title": "Système de management de la qualité et ses processus",
            "keywords": ["smq", "processus", "système", "management", "approche processus"]
        },
        "5.1": {
            "title": "Leadership et engagement",
            "keywords": ["leadership", "direction", "engagement", "responsabilité", "politique"]
        },
        "5.2": {
            "title": "Politique qualité",
            "keywords": ["politique", "qualité", "objectifs", "engagement", "amélioration"]
        },
        "5.3": {
            "title": "Rôles, responsabilités et autorités",
            "keywords": ["rôles", "responsabilités", "autorités", "organigramme", "fonction"]
        },
        "6.1": {
            "title": "Actions face aux risques et opportunités",
            "keywords": ["risques", "opportunités", "planification", "prévention", "mitigation"]
        },
        "6.2": {
            "title": "Objectifs qualité et planification",
            "keywords": ["objectifs", "planification", "mesurable", "surveillance", "indicateurs"]
        },
        "6.3": {
            "title": "Planification des modifications",
            "keywords": ["modifications", "changements", "planification", "contrôle"]
        },
        "7.1.1": {
            "title": "Ressources - Généralités",
            "keywords": ["ressources", "personnes", "infrastructure", "environnement"]
        },
        "7.1.2": {
            "title": "Ressources humaines",
            "keywords": ["personnel", "compétence", "formation", "qualification"]
        },
        "7.1.3": {
            "title": "Infrastructure",
            "keywords": ["infrastructure", "équipements", "installations", "maintenance"]
        },
        "7.1.4": {
            "title": "Environnement pour la mise en œuvre des processus",
            "keywords": ["environnement", "conditions", "travail", "locaux"]
        },
        "7.1.5": {
            "title": "Ressources pour la surveillance et la mesure",
            "keywords": ["surveillance", "mesure", "équipements", "étalonnage", "vérification"]
        },
        "7.1.6": {
            "title": "Connaissances organisationnelles",
            "keywords": ["connaissances", "savoir", "expertise", "capitalisation"]
        },
        "7.2": {
            "title": "Compétences",
            "keywords": ["compétences", "formation", "qualification", "habilitation"]
        },
        "7.3": {
            "title": "Sensibilisation",
            "keywords": ["sensibilisation", "conscience", "communication", "information"]
        },
        "7.4": {
            "title": "Communication",
            "keywords": ["communication", "information", "échange", "diffusion"]
        },
        "7.5": {
            "title": "Informations documentées",
            "keywords": ["documents", "enregistrements", "procédures", "instructions", "formulaires"]
        },
        "8.1": {
            "title": "Planification et maîtrise opérationnelles",
            "keywords": ["planification", "opérations", "production", "réalisation", "maîtrise"]
        },
        "8.2": {
            "title": "Exigences relatives aux produits et services",
            "keywords": ["exigences", "clients", "produits", "services", "spécifications"]
        },
        "8.3": {
            "title": "Conception et développement",
            "keywords": ["conception", "développement", "design", "validation", "prototype"]
        },
        "8.4": {
            "title": "Maîtrise des processus, produits et services fournis par des prestataires externes",
            "keywords": ["fournisseurs", "sous-traitants", "achats", "prestataires", "externes"]
        },
        "8.5": {
            "title": "Production et prestation de service",
            "keywords": ["production", "fabrication", "assemblage", "prestation", "réalisation"]
        },
        "8.5.1": {
            "title": "Maîtrise de la production et de la prestation de service",
            "keywords": ["maîtrise", "production", "fabrication", "contrôle", "surveillance"]
        },
        "8.5.2": {
            "title": "Identification et traçabilité",
            "keywords": ["identification", "traçabilité", "numéro", "série", "lot", "marquage"]
        },
        "8.5.3": {
            "title": "Propriété des clients ou des prestataires externes",
            "keywords": ["propriété", "client", "fourniture", "protection"]
        },
        "8.5.4": {
            "title": "Préservation",
            "keywords": ["préservation", "stockage", "protection", "conditionnement", "emballage"]
        },
        "8.5.5": {
            "title": "Activités après livraison",
            "keywords": ["après-vente", "garantie", "maintenance", "sav", "support"]
        },
        "8.5.6": {
            "title": "Maîtrise des modifications",
            "keywords": ["modifications", "changements", "revue", "approbation"]
        },
        "8.6": {
            "title": "Libération des produits et services",
            "keywords": ["libération", "acceptation", "contrôle final", "validation", "conformité"]
        },
        "8.7": {
            "title": "Maîtrise des éléments de sortie non conformes",
            "keywords": ["non-conformité", "rebut", "retouche", "dérogation", "nc"]
        },
        "9.1": {
            "title": "Surveillance, mesure, analyse et évaluation",
            "keywords": ["surveillance", "mesure", "analyse", "kpi", "indicateurs", "performance"]
        },
        "9.1.1": {
            "title": "Généralités - Surveillance et mesure",
            "keywords": ["mesure", "surveillance", "suivi", "contrôle"]
        },
        "9.1.2": {
            "title": "Satisfaction du client",
            "keywords": ["satisfaction", "client", "enquête", "feedback", "réclamation"]
        },
        "9.1.3": {
            "title": "Analyse et évaluation",
            "keywords": ["analyse", "évaluation", "données", "tendances", "performance"]
        },
        "9.2": {
            "title": "Audit interne",
            "keywords": ["audit", "interne", "vérification", "évaluation", "conformité"]
        },
        "9.3": {
            "title": "Revue de direction",
            "keywords": ["revue", "direction", "management", "bilan", "décision"]
        },
        "10.1": {
            "title": "Généralités - Amélioration",
            "keywords": ["amélioration", "continue", "progrès", "optimisation"]
        },
        "10.2": {
            "title": "Non-conformité et action corrective",
            "keywords": ["non-conformité", "action corrective", "correction", "nc", "traitement"]
        },
        "10.3": {
            "title": "Amélioration continue",
            "keywords": ["amélioration continue", "kaizen", "pdca", "innovation"]
        }
    }
    
    def __init__(self, llm: ChatGroq, rag_system):
        """
        Initialise l'agent mappeur
        
        Args:
            llm: Modèle de langage Groq
            rag_system: Système RAG pour récupérer les exigences ISO
        """
        self.llm = llm
        self.rag_system = rag_system
        self.mapping_prompt = self._create_mapping_prompt()
    
    def _create_mapping_prompt(self) -> PromptTemplate:
        """Créer le prompt pour le mapping de clauses"""
        template = """Tu es un expert ISO 9001 spécialisé dans le mapping de documents.

SECTION DU DOCUMENT À ANALYSER:
Titre: {section_title}
Type: {section_type}
Contenu: {section_content}

TÂCHE:
Identifie les clauses ISO 9001:2015 qui sont pertinentes pour cette section.
Pour chaque clause identifiée, fournis:
1. Le numéro de clause (ex: "8.5.1")
2. Un score de pertinence (0.0 à 1.0)
3. Une justification courte

CLAUSES ISO 9001 DISPONIBLES:
{available_clauses}

INSTRUCTIONS:
- Identifie 2 à 5 clauses maximum les plus pertinentes
- Score > 0.7 = très pertinent, 0.5-0.7 = pertinent, < 0.5 = peu pertinent
- Base-toi sur le contenu et le contexte de la section
- Sois précis et concis dans les justifications

RÉPONDS UNIQUEMENT EN FORMAT JSON (sans markdown, sans backticks):
{{
  "mapped_clauses": [
    {{
      "clause_number": "X.X",
      "relevance_score": 0.0,
      "justification": "..."
    }}
  ]
}}"""
        
        return PromptTemplate(
            template=template,
            input_variables=["section_title", "section_type", "section_content", "available_clauses"]
        )
    
    def map_section_to_clauses(self, section: DocumentSection, max_clauses: int = 5) -> List[ISOClauseMapping]:
        """
        Mapper une section aux clauses ISO pertinentes
        
        Args:
            section: Section du document à analyser
            max_clauses: Nombre maximum de clauses à retourner
            
        Returns:
            Liste de mappings clause ISO
        """
        logger.info(f"Mapping de la section: {section.title}")
        
        # Étape 1: Pré-filtrage par mots-clés
        candidate_clauses = self._prefilter_clauses_by_keywords(section)
        
        # Étape 2: Utiliser le LLM pour affiner et scorer
        mappings = self._llm_mapping(section, candidate_clauses)
        
        # Étape 3: Enrichir avec les exigences ISO via RAG
        enriched_mappings = self._enrich_with_iso_requirements(mappings)
        
        # Trier par score de pertinence et limiter
        enriched_mappings.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return enriched_mappings[:max_clauses]
    
    def _prefilter_clauses_by_keywords(self, section: DocumentSection) -> List[str]:
        """Pré-filtrer les clauses par analyse de mots-clés"""
        section_text = (section.title + " " + section.content).lower()
        
        matched_clauses = []
        
        for clause_num, clause_info in self.ISO_9001_CLAUSE_MAP.items():
            keywords = clause_info["keywords"]
            
            # Compter les mots-clés présents
            matches = sum(1 for keyword in keywords if keyword in section_text)
            
            if matches > 0:
                matched_clauses.append(clause_num)
        
        logger.debug(f"Pré-filtrage: {len(matched_clauses)} clauses candidates")
        return matched_clauses[:10]  # Limiter à 10 pour le LLM
    
    def _llm_mapping(self, section: DocumentSection, candidate_clauses: List[str]) -> List[ISOClauseMapping]:
        """Utiliser le LLM pour mapper et scorer les clauses"""
        
        # Préparer la liste des clauses disponibles
        available_clauses_text = "\n".join([
            f"- {num}: {self.ISO_9001_CLAUSE_MAP[num]['title']}"
            for num in candidate_clauses
        ])
        
        # Limiter le contenu de la section pour le prompt
        section_content = section.content[:1500] if len(section.content) > 1500 else section.content
        
        try:
            # Appeler le LLM
            prompt = self.mapping_prompt.format(
                section_title=section.title,
                section_type=section.section_type or "general",
                section_content=section_content,
                available_clauses=available_clauses_text
            )
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Parser la réponse JSON
            # Nettoyer les backticks markdown si présents
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # Créer les objets ISOClauseMapping
            mappings = []
            for clause_data in result.get("mapped_clauses", []):
                clause_num = clause_data["clause_number"]
                
                if clause_num in self.ISO_9001_CLAUSE_MAP:
                    mapping = ISOClauseMapping(
                        clause_number=clause_num,
                        clause_title=self.ISO_9001_CLAUSE_MAP[clause_num]["title"],
                        relevance_score=float(clause_data["relevance_score"]),
                        justification=clause_data["justification"],
                        iso_requirements="",  # Sera enrichi plus tard
                        source_document="ISO 9001:2015"
                    )
                    mappings.append(mapping)
            
            return mappings
            
        except Exception as e:
            logger.error(f"Erreur lors du mapping LLM: {e}")
            # Fallback: retourner les clauses candidates avec score moyen
            return [
                ISOClauseMapping(
                    clause_number=clause_num,
                    clause_title=self.ISO_9001_CLAUSE_MAP[clause_num]["title"],
                    relevance_score=0.6,
                    justification="Mapping automatique par mots-clés",
                    iso_requirements="",
                    source_document="ISO 9001:2015"
                )
                for clause_num in candidate_clauses[:5]
            ]
    
    def _enrich_with_iso_requirements(self, mappings: List[ISOClauseMapping]) -> List[ISOClauseMapping]:
        """Enrichir les mappings avec les exigences ISO depuis le RAG"""
        
        for mapping in mappings:
            try:
                # Rechercher les exigences de la clause dans le RAG
                query = f"Quelles sont les exigences de la clause {mapping.clause_number} de la norme ISO 9001:2015?"
                
                results = self.rag_system.search_documents(query, k=2)
                
                if results:
                    # Combiner les contenus trouvés
                    requirements = "\n\n".join([
                        result['content'] for result in results
                        if result.get('section', '') == mapping.clause_number
                    ])
                    
                    if requirements:
                        mapping.iso_requirements = requirements[:500]  # Limiter la taille
                    else:
                        # Prendre le premier résultat même s'il ne correspond pas exactement
                        mapping.iso_requirements = results[0]['content'][:500]
                
            except Exception as e:
                logger.warning(f"Impossible d'enrichir la clause {mapping.clause_number}: {e}")
        
        return mappings


# Fonction utilitaire pour tester
def test_clause_mapper(rag_system, test_section: DocumentSection):
    """Tester le mappeur de clauses"""
    from langchain_groq import ChatGroq
    import os
    
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )
    
    mapper = ClauseMappingAgent(llm, rag_system)
    mappings = mapper.map_section_to_clauses(test_section)
    
    print(f"\n=== Mappings pour: {test_section.title} ===")
    for mapping in mappings:
        print(f"\nClause {mapping.clause_number}: {mapping.clause_title}")
        print(f"  Pertinence: {mapping.relevance_score:.2f}")
        print(f"  Justification: {mapping.justification}")
        print(f"  Exigences ISO: {mapping.iso_requirements[:200]}...")