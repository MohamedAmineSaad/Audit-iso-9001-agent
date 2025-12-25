"""
Système RAG ISO 9001/9000 avec Analyse Agentique
Application principale complète pour indexation, interrogation et analyse de conformité
"""

import sys
from pathlib import Path
import argparse
import logging
from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.document_processor import ISO9001DocumentProcessor
from src.vectorstore_manager import ChromaDBManager
from src.rag_system import ISO9001RAGSystem
from src.document_extractor import DocumentExtractor, DocumentSection
from src.conformity_analyzer import (
    ClauseMappingAgent, 
    ConformityVerificationAgent,
    SectionAnalysisResult,
    ConformityStatus
)

# Import agentic system
from src.iso_agentic_graph import ISOAgenticAnalyzer
from src.iso_agentic_state import ConfidenceLevel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()


class ISO9001RAGApp:
    """Application principale du système RAG ISO 9001 avec analyse agentique"""
    
    def __init__(self):
        # Valider la configuration
        Config.validate_config()
        
        # Initialiser les composants principaux
        self.doc_processor = ISO9001DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        self.vectorstore_manager = ChromaDBManager(
            persist_directory=Config.VECTORSTORE_DIR,
            collection_name=Config.COLLECTION_NAME,
            embedding_model=Config.EMBEDDING_MODEL
        )
        
        self.rag_system = ISO9001RAGSystem(
            groq_api_key=Config.GROQ_API_KEY,
            vectorstore_manager=self.vectorstore_manager,
            model_name=Config.GROQ_MODEL,
            temperature=Config.GROQ_TEMPERATURE
        )
        
        # Extracteur de documents
        self.doc_extractor = DocumentExtractor()
        
        # Agents d'analyse (lazy loading)
        self.clause_mapper = None
        self.conformity_verifier = None
        
        # Analyseur agentique (lazy loading)
        self.agentic_analyzer = None
    
    # ========================================================================
    # INDEXATION
    # ========================================================================
    
    def index_documents(self, force_reindex: bool = False):
        """Indexer les documents ISO dans ChromaDB"""
        console.print("\n[bold blue]📄 INDEXATION DES DOCUMENTS ISO[/bold blue]\n")
        
        # Vérifier si déjà indexé
        if not force_reindex and not self.vectorstore_manager.check_if_empty():
            stats = self.vectorstore_manager.get_collection_stats()
            console.print(f"[yellow]ℹ️  Base de données déjà indexée ({stats['total_documents']} documents)[/yellow]")
            
            response = input("Voulez-vous réindexer ? (o/n): ")
            if response.lower() != 'o':
                console.print("[green]✓ Utilisation de l'index existant[/green]")
                return
            
            self.vectorstore_manager.reset_vectorstore()
        
        # Préparer les chemins des PDFs
        pdf_paths = {
            name: Config.RAW_DATA_DIR / filename
            for name, filename in Config.SOURCE_DOCS.items()
        }
        
        # Vérifier l'existence des fichiers
        missing_files = [name for name, path in pdf_paths.items() if not path.exists()]
        if missing_files:
            console.print(f"[red]❌ Fichiers manquants: {', '.join(missing_files)}[/red]")
            console.print(f"[yellow]Placer les PDFs dans: {Config.RAW_DATA_DIR}[/yellow]")
            return
        
        # Traiter les documents
        console.print("[cyan]📄 Traitement des documents PDF...[/cyan]")
        chunks = self.doc_processor.process_iso_documents(pdf_paths)
        
        if not chunks:
            console.print("[red]❌ Aucun document traité[/red]")
            return
        
        # Afficher les statistiques
        stats = self.doc_processor.get_statistics(chunks)
        self._display_processing_stats(stats)
        
        # Indexer dans ChromaDB
        console.print("\n[cyan]💾 Indexation dans ChromaDB...[/cyan]")
        self.vectorstore_manager.add_documents(chunks)
        
        # Statistiques finales
        final_stats = self.vectorstore_manager.get_collection_stats()
        console.print(f"\n[bold green]✓ Indexation terminée![/bold green]")
        console.print(f"Total de documents indexés: {final_stats['total_documents']}")
    
    def _display_processing_stats(self, stats: dict):
        """Afficher les statistiques de traitement"""
        table = Table(title="Statistiques de traitement")
        table.add_column("Métrique", style="cyan")
        table.add_column("Valeur", style="green")
        
        table.add_row("Total de chunks", str(stats['total_chunks']))
        table.add_row("Longueur moyenne", f"{stats['avg_length']:.0f} caractères")
        table.add_row("Sections uniques", str(stats['unique_sections']))
        
        for source, count in stats['sources'].items():
            table.add_row(f"Chunks de {source}", str(count))
        
        console.print(table)
    
    # ========================================================================
    # MODE CHAT INTERACTIF
    # ========================================================================
    
    def interactive_mode(self):
        """Mode interactif pour poser des questions"""
        console.print("\n[bold green]💬 MODE INTERACTIF ISO 9001[/bold green]")
        console.print("[dim]Tapez 'exit' pour quitter, 'help' pour l'aide[/dim]\n")
        
        # Vérifier que la base est indexée
        if self.vectorstore_manager.check_if_empty():
            console.print("[red]❌ Base de données ISO non indexée![/red]")
            console.print("[yellow]Exécutez d'abord: python main.py index[/yellow]")
            return
        
        # Initialiser le système
        self.rag_system.initialize_qa_chain()
        
        while True:
            try:
                question = input("\n❓ Votre question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() == 'exit':
                    console.print("[yellow]👋 Au revoir![/yellow]")
                    break
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if question.lower().startswith('section:'):
                    section = question.split(':', 1)[1].strip()
                    self._explain_section(section)
                    continue
                
                # Poser la question
                console.print("\n[cyan]🤔 Recherche en cours...[/cyan]")
                result = self.rag_system.ask(question)
                
                # Afficher la réponse
                self._display_answer(result)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Au revoir![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]❌ Erreur: {e}[/red]")
                logger.exception("Erreur dans le mode interactif")
    
    def _display_answer(self, result: dict):
        """Afficher une réponse formatée"""
        # Afficher la réponse
        console.print(Panel(
            Markdown(result['answer']),
            title="[bold green]📋 Réponse[/bold green]",
            border_style="green"
        ))
        
        # Afficher les sources
        if result['sources']:
            console.print("\n[bold cyan]📚 Sources:[/bold cyan]")
            for source in result['sources']:
                console.print(f"\n  [{source['rank']}] {source['source']} - Section {source['section']}")
                console.print(f"      Page {source['page']} | {source['section_title']}")
                console.print(f"      [dim]{source['content_preview']}[/dim]")
    
    def _explain_section(self, section_number: str):
        """Expliquer une section spécifique"""
        console.print(f"\n[cyan]📖 Explication de la section {section_number}...[/cyan]")
        result = self.rag_system.explain_section(section_number)
        self._display_answer(result)
    
    def _show_help(self):
        """Afficher l'aide"""
        help_text = """
        **Commandes disponibles:**
        
        - Posez directement vos questions sur ISO 9001/9000
        - `section:X.X` - Expliquer une section spécifique (ex: section:4.1)
        - `help` - Afficher cette aide
        - `exit` - Quitter le programme
        
        **Exemples de questions:**
        - Quelles sont les exigences de la section 4.1 ?
        - Comment mettre en place un système de management de la qualité ?
        - Quelle est la différence entre action corrective et action préventive ?
        - Expliquer le concept d'approche processus
        """
        console.print(Panel(Markdown(help_text), title="Aide", border_style="blue"))
    
    # ========================================================================
    # MODE RECHERCHE
    # ========================================================================
    
    def search_mode(self, query: str, k: int = 5):
        """Mode recherche simple"""
        console.print(f"\n[cyan]🔍 Recherche: {query}[/cyan]\n")
        
        # Vérifier que la base est indexée
        if self.vectorstore_manager.check_if_empty():
            console.print("[red]❌ Base de données ISO non indexée![/red]")
            console.print("[yellow]Exécutez d'abord: python main.py index[/yellow]")
            return
        
        results = self.rag_system.search_documents(query, k=k)
        
        if not results:
            console.print("[yellow]Aucun résultat trouvé[/yellow]")
            return
        
        for i, result in enumerate(results, 1):
            console.print(f"\n[bold]{i}. {result['source']}[/bold] (Similarité: {result['similarity_score']:.2%})")
            console.print(f"   Page {result['page']} | Section {result['section']}")
            console.print(f"   [dim]{result['content'][:300]}...[/dim]")
    
    # ========================================================================
    # MODE STATISTIQUES
    # ========================================================================
    
    def stats_mode(self):
        """Afficher les statistiques du système"""
        stats = self.vectorstore_manager.get_collection_stats()
        
        console.print("\n[bold cyan]📊 STATISTIQUES DU SYSTÈME[/bold cyan]\n")
        
        table = Table()
        table.add_column("Paramètre", style="cyan")
        table.add_column("Valeur", style="green")
        
        table.add_row("Collection", stats.get('collection_name', 'N/A'))
        table.add_row("Documents indexés", str(stats.get('total_documents', 0)))
        table.add_row("Modèle d'embeddings", stats.get('embedding_model', 'N/A'))
        table.add_row("Modèle LLM", Config.GROQ_MODEL)
        table.add_row("Répertoire", str(stats.get('persist_directory', 'N/A')))
        
        console.print(table)
    
    # ========================================================================
    # ANALYSE DE CONFORMITÉ STANDARD
    # ========================================================================
    
    def analyze_document_conformity(self, document_path: str):
        """
        Analyser la conformité d'un document d'entreprise (méthode standard)
        
        Args:
            document_path: Chemin vers le document à analyser
        """
        console.print("\n[bold blue]🔍 ANALYSE DE CONFORMITÉ ISO 9001/9000[/bold blue]\n")
        
        # Vérifier que le document existe (gestion robuste des chemins)
        base_dir = Path(__file__).parent
        doc_path_input = Path(document_path)
        candidates = []
        # 1) Chemin tel que fourni (relatif au CWD ou absolu)
        candidates.append(doc_path_input)
        # 2) Relatif au dossier du script
        candidates.append(base_dir / doc_path_input)
        # 3) Si l'utilisateur a préfixé par le nom du projet, enlever ce préfixe
        parts = list(doc_path_input.parts)
        if parts and parts[0] == base_dir.name:
            candidates.append(base_dir / Path(*parts[1:]))
        
        doc_path = None
        for cand in candidates:
            if cand.exists():
                doc_path = cand
                break
        
        if doc_path is None:
            console.print(f"[red]❌ Fichier introuvable: {document_path}[/red]")
            console.print("[yellow]Astuce: utilisez un chemin relatif depuis le projet, par ex.: Doc/formulaire deepseek favrication.docx[/yellow]")
            return
        
        # Vérifier que le RAG est initialisé
        if self.vectorstore_manager.check_if_empty():
            console.print("[red]❌ Base de données ISO non indexée![/red]")
            console.print("[yellow]Exécutez d'abord: python main.py index[/yellow]")
            return
        
        try:
            # Étape 1 : Extraction et segmentation
            console.print(f"[cyan]📄 Extraction du document: {doc_path.name}[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Extraction en cours...", total=None)
                sections, metadata = self.doc_extractor.extract_and_segment(str(doc_path))
                progress.update(task, completed=True)
            
            console.print(f"[green]✓ Document segmenté en {len(sections)} sections[/green]\n")
            
            # Afficher les sections détectées
            self._display_sections_summary(sections, metadata)
            
            # Étape 2 : Analyse avec mapping de clauses
            console.print("\n[cyan]🔍 Mapping des clauses ISO...[/cyan]\n")
            analysis_results = self._map_clauses_for_sections(sections)
            
            # Afficher les résultats du mapping
            self._display_clause_mappings(analysis_results)
            
            # Étape 3 : Vérification de conformité
            console.print("\n[cyan]✅ Vérification de conformité...[/cyan]\n")
            analysis_results = self._verify_conformity_for_sections(sections, analysis_results)
            
            # Afficher les résultats de conformité
            self._display_conformity_results(analysis_results)
            
            # Étape 4 : Génération du rapport final
            console.print("\n[bold yellow]📋 Génération du rapport final...[/bold yellow]")
            self._generate_summary_report(analysis_results)
            
        except Exception as e:
            console.print(f"[red]❌ Erreur lors de l'analyse: {e}[/red]")
            logger.exception("Erreur d'analyse")
    
    def _map_clauses_for_sections(self, sections: list) -> List[SectionAnalysisResult]:
        """Mapper les clauses ISO pour les sections clés du document"""
        # Initialiser le mappeur si nécessaire
        if self.clause_mapper is None:
            self.clause_mapper = ClauseMappingAgent(
                llm=self.rag_system.llm,
                rag_system=self.rag_system
            )
        
        # Sélectionner les sections les plus importantes
        key_sections = self._select_key_sections(sections)
        
        console.print(f"[bold]Analyse de {len(key_sections)} sections clés:[/bold]\n")
        
        analysis_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            for section in key_sections:
                task = progress.add_task(f"Mapping: {section.title[:40]}...", total=None)
                
                try:
                    # Mapper la section aux clauses ISO
                    mappings = self.clause_mapper.map_section_to_clauses(section, max_clauses=5)
                    
                    # Créer le résultat d'analyse
                    result = SectionAnalysisResult(
                        section_title=section.title,
                        section_type=section.section_type or "general",
                        mapped_clauses=mappings
                    )
                    
                    analysis_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Erreur lors du mapping de '{section.title}': {e}")
                
                progress.update(task, completed=True)
        
        return analysis_results
    
    def _select_key_sections(self, sections: list) -> list:
        """Sélectionner les sections les plus importantes à analyser"""
        # Prioriser les formulaires et procédures
        priority_sections = [
            s for s in sections 
            if s.section_type in ['formulaire', 'procedure', 'instruction']
        ]
        
        # Si pas assez, ajouter des sections générales avec du contenu substantiel
        if len(priority_sections) < 5:
            other_sections = [
                s for s in sections 
                if s.section_type == 'general' and len(s.content) > 300
            ]
            priority_sections.extend(other_sections)
        
        # Limiter à 8 sections max pour ne pas surcharger
        return priority_sections[:8]
    
    def _display_clause_mappings(self, analysis_results: List[SectionAnalysisResult]):
        """Afficher les résultats du mapping de clauses"""
        
        for result in analysis_results:
            console.print(f"\n[bold cyan]📌 {result.section_title}[/bold cyan]")
            console.print(f"   Type: [yellow]{result.section_type}[/yellow]\n")
            
            if not result.mapped_clauses:
                console.print("   [dim]Aucune clause ISO identifiée[/dim]")
                continue
            
            # Afficher chaque clause avec détails complets
            for i, mapping in enumerate(result.mapped_clauses, 1):
                # Barre de score visuelle
                score_pct = int(mapping.relevance_score * 100)
                if score_pct >= 80:
                    score_color = "green"
                    score_icon = "✓"
                elif score_pct >= 60:
                    score_color = "yellow"
                    score_icon = "●"
                else:
                    score_color = "red"
                    score_icon = "⚠"
                
                # Affichage formaté
                console.print(f"   [{score_color}]{score_icon} Clause {mapping.clause_number}[/{score_color}] - [{score_color}]{score_pct}%[/{score_color}]")
                console.print(f"      [bold]{mapping.clause_title}[/bold]")
                console.print(f"      [dim]Justification:[/dim] {mapping.justification}")
                
                # Afficher un extrait des exigences ISO si disponibles
                if mapping.iso_requirements:
                    console.print(f"      [dim]Exigences ISO:[/dim] {mapping.iso_requirements[:150]}...")
                
                if i < len(result.mapped_clauses):
                    console.print()  # Ligne vide entre les clauses
    
    def _verify_conformity_for_sections(self, sections: list, analysis_results: List[SectionAnalysisResult]) -> List[SectionAnalysisResult]:
        """Vérifier la conformité pour chaque section analysée"""
        # Initialiser le vérificateur si nécessaire
        if self.conformity_verifier is None:
            self.conformity_verifier = ConformityVerificationAgent(
                llm=self.rag_system.llm,
                rag_system=self.rag_system
            )
        
        # Créer un mapping section_title -> section
        sections_dict = {s.title: s for s in sections}
        
        console.print(f"[bold]Vérification de conformité pour {len(analysis_results)} sections:[/bold]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            for result in analysis_results:
                task = progress.add_task(f"Vérification: {result.section_title[:40]}...", total=None)
                
                try:
                    # Récupérer la section correspondante
                    section = sections_dict.get(result.section_title)
                    
                    if section and result.mapped_clauses:
                        # Vérifier la conformité pour chaque clause mappée
                        conformity_analyses = self.conformity_verifier.verify_multiple_clauses(
                            section, 
                            result.mapped_clauses
                        )
                        
                        result.conformity_analyses = conformity_analyses
                        
                        # Calculer le score global de la section
                        if conformity_analyses:
                            result.overall_score = sum(
                                a.conformity_score for a in conformity_analyses
                            ) / len(conformity_analyses)
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la vérification de '{result.section_title}': {e}")
                
                progress.update(task, completed=True)
        
        return analysis_results
    
    def _display_conformity_results(self, analysis_results: List[SectionAnalysisResult]):
        """Afficher les résultats de conformité détaillés"""
        
        for result in analysis_results:
            if not result.conformity_analyses:
                continue
            
            console.print(f"\n[bold cyan]📊 Résultats de conformité: {result.section_title}[/bold cyan]")
            console.print(f"   Score global: [bold]{result.overall_score:.0f}%[/bold]\n")
            
            for analysis in result.conformity_analyses:
                # Icône selon le statut
                if analysis.status == ConformityStatus.CONFORME:
                    status_icon = "✅"
                    status_color = "green"
                elif analysis.status == ConformityStatus.PARTIELLEMENT_CONFORME:
                    status_icon = "⚠️"
                    status_color = "yellow"
                elif analysis.status == ConformityStatus.NON_CONFORME:
                    status_icon = "❌"
                    status_color = "red"
                else:
                    status_icon = "❓"
                    status_color = "dim"
                
                console.print(f"   {status_icon} [bold]Clause {analysis.clause_number}[/bold] - [{status_color}]{analysis.conformity_score}%[/{status_color}]")
                console.print(f"      {analysis.clause_title}")
                console.print(f"      Statut: [{status_color}]{analysis.status.value}[/{status_color}]")
                
                # Éléments conformes
                if analysis.conformity_elements:
                    console.print(f"\n      [green]✓ Éléments conformes:[/green]")
                    for elem in analysis.conformity_elements[:3]:  # Limiter à 3
                        console.print(f"        • {elem}")
                
                # Non-conformités
                if analysis.non_conformities:
                    console.print(f"\n      [red]✗ Non-conformités détectées:[/red]")
                    for nc in analysis.non_conformities[:3]:  # Limiter à 3
                        console.print(f"        • {nc}")
                
                # Recommandations
                if analysis.recommendations:
                    console.print(f"\n      [yellow]💡 Recommandations:[/yellow]")
                    for rec in analysis.recommendations[:2]:  # Limiter à 2
                        console.print(f"        • {rec}")
                
                console.print()  # Ligne vide entre les analyses
    
    def _generate_summary_report(self, analysis_results: List[SectionAnalysisResult]):
        """Générer un rapport récapitulatif"""
        console.print("\n[bold blue]📋 RAPPORT RÉCAPITULATIF[/bold blue]\n")
        
        # Calculer les statistiques globales
        total_clauses = sum(len(r.conformity_analyses) for r in analysis_results)
        
        if total_clauses == 0:
            console.print("[yellow]Aucune clause analysée[/yellow]")
            return
        
        # Compter par statut
        status_counts = {
            ConformityStatus.CONFORME: 0,
            ConformityStatus.PARTIELLEMENT_CONFORME: 0,
            ConformityStatus.NON_CONFORME: 0,
            ConformityStatus.NON_APPLICABLE: 0
        }
        
        total_score = 0
        score_count = 0
        
        for result in analysis_results:
            for analysis in result.conformity_analyses:
                status_counts[analysis.status] = status_counts.get(analysis.status, 0) + 1
                total_score += analysis.conformity_score
                score_count += 1
        
        # Score global
        global_score = total_score / score_count if score_count > 0 else 0
        
        # Afficher le tableau récapitulatif
        summary_table = Table(title="Synthèse de l'analyse")
        summary_table.add_column("Métrique", style="cyan")
        summary_table.add_column("Valeur", style="green")
        
        summary_table.add_row("Score de conformité global", f"{global_score:.1f}%")
        summary_table.add_row("Total de clauses analysées", str(total_clauses))
        summary_table.add_row("Clauses conformes", f"[green]{status_counts.get(ConformityStatus.CONFORME, 0)}[/green]")
        summary_table.add_row("Clauses partiellement conformes", f"[yellow]{status_counts.get(ConformityStatus.PARTIELLEMENT_CONFORME, 0)}[/yellow]")
        summary_table.add_row("Clauses non conformes", f"[red]{status_counts.get(ConformityStatus.NON_CONFORME, 0)}[/red]")
        
        console.print(summary_table)
        
        # Recommandations prioritaires
        all_recommendations = []
        for result in analysis_results:
            for analysis in result.conformity_analyses:
                if analysis.status in [ConformityStatus.NON_CONFORME, ConformityStatus.PARTIELLEMENT_CONFORME]:
                    for rec in analysis.recommendations:
                        all_recommendations.append((analysis.clause_number, rec))
        
        if all_recommendations:
            console.print("\n[bold yellow]🎯 Actions prioritaires:[/bold yellow]")
            for clause, rec in all_recommendations[:5]:  # Top 5
                console.print(f"  • [Clause {clause}] {rec}")
    
    def _display_sections_summary(self, sections: list, metadata: dict):
        """Afficher un résumé des sections détectées"""
        # Métadonnées du document
        meta_table = Table(title="📊 Métadonnées du document")
        meta_table.add_column("Propriété", style="cyan")
        meta_table.add_column("Valeur", style="green")
        
        meta_table.add_row("Nom du fichier", metadata.get('file_name', 'N/A'))
        meta_table.add_row("Type", metadata.get('file_type', 'N/A').upper())
        meta_table.add_row("Nombre de sections", str(len(sections)))
        
        if 'pages' in metadata:
            meta_table.add_row("Nombre de pages", str(metadata['pages']))
        
        console.print(meta_table)
        
        # Sections détectées
        sections_table = Table(title="\n📑 Sections détectées")
        sections_table.add_column("#", style="cyan", width=4)
        sections_table.add_column("Titre", style="white")
        sections_table.add_column("Type", style="yellow", width=15)
        sections_table.add_column("Mots-clés ISO", style="green")
        sections_table.add_column("Taille", style="blue", width=10)
        
        for i, section in enumerate(sections, 1):
            keywords = ", ".join(section.keywords[:3]) if section.keywords else "-"
            if len(section.keywords) > 3:
                keywords += f" +{len(section.keywords)-3}"
            
            sections_table.add_row(
                str(i),
                section.title[:50] + ("..." if len(section.title) > 50 else ""),
                section.section_type or "général",
                keywords,
                f"{len(section.content)} car."
            )
        
        console.print(sections_table)
    
    # ========================================================================
    # ANALYSE AGENTIQUE (NOUVEAU)
    # ========================================================================
    
    def _initialize_agentic_analyzer(self):
        """Initialiser l'analyseur agentique (lazy loading)"""
        if self.agentic_analyzer is None:
            console.print("[cyan]🧠 Initialisation de l'analyseur agentique...[/cyan]")
            
            self.agentic_analyzer = ISOAgenticAnalyzer(
                vectorstore_manager=self.vectorstore_manager,
                rag_system=self.rag_system
            )
            
            console.print("[green]✓ Analyseur agentique prêt[/green]")
    
    def analyze_document_conformity_agentic(self, document_path: str):
        """
        Analyser la conformité d'un document avec le workflow AGENTIQUE
        
        Cette version utilise l'intelligence agentique pour:
        - Évaluer la qualité de la récupération
        - Décider d'élargir le contexte si nécessaire
        - Vérifier les hallucinations
        - Calculer un score de confiance
        """
        console.print("\n[bold blue]🤖 ANALYSE AGENTIQUE ISO 9001/9000[/bold blue]\n")
        
        # Vérifier que le document existe
        doc_path = Path(document_path)
        if not doc_path.exists():
            console.print(f"[red]❌ Fichier introuvable: {doc_path}[/red]")
            return
        
        # Vérifier que le RAG est initialisé
        if self.vectorstore_manager.check_if_empty():
            console.print("[red]❌ Base de données ISO non indexée![/red]")
            console.print("[yellow]Exécutez d'abord: python main.py index[/yellow]")
            return
        
        # Initialiser l'analyseur agentique
        self._initialize_agentic_analyzer()
        
        try:
            # Étape 1 : Extraction du document
            console.print(f"[cyan]📄 Extraction: {doc_path.name}[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Traitement du document...", total=None)
                sections, metadata = self.doc_extractor.extract_and_segment(str(doc_path))
                progress.update(task, completed=True)
            
            console.print(f"[green]✓ Extrait {len(sections)} sections[/green]\n")
            
            # Afficher métadonnées
            self._display_document_metadata(metadata, len(sections))
            
            # Étape 2 : Sélection des sections clés
            key_sections = self._select_key_sections(sections)
            
            console.print(f"\n[bold]Analyse agentique de {len(key_sections)} sections prioritaires:[/bold]\n")
            
            # Étape 3 : Analyse agentique pour chaque section
            all_results = []
            
            for i, section in enumerate(key_sections, 1):
                console.print(f"\n[bold cyan]Section {i}/{len(key_sections)}: {section.title}[/bold cyan]")
                
                # Exécuter l'analyse agentique
                result = self.agentic_analyzer.analyze_section(section)
                all_results.append(result)
                
                # Afficher les résultats
                self._display_agentic_results(result)
            
            # Étape 4 : Rapport global
            console.print("\n[bold blue]📊 RAPPORT GLOBAL[/bold blue]\n")
            self._generate_global_report(all_results)
            
        except Exception as e:
            console.print(f"[red]❌ Erreur: {e}[/red]")
            logger.exception("Erreur d'analyse agentique")
    
    def _display_document_metadata(self, metadata: dict, num_sections: int):
        """Afficher les métadonnées du document"""
        table = Table(title="📄 Informations du document")
        table.add_column("Propriété", style="cyan")
        table.add_column("Valeur", style="green")
        
        table.add_row("Nom du fichier", metadata.get("file_name", "N/A"))
        table.add_row("Type", metadata.get("file_type", "N/A").upper())
        table.add_row("Sections", str(num_sections))
        
        if "pages" in metadata:
            table.add_row("Pages", str(metadata["pages"]))
        
        console.print(table)
    
    def _display_agentic_results(self, result: dict):
        """
        Afficher les résultats de l'analyse agentique
        
        Montre: confiance, risques, scores de conformité, indicateurs de qualité
        """
        
        # Score de confiance
        confidence = result.get("confidence_score")
        if confidence:
            # Couleur basée sur le niveau de confiance
            if confidence.level == ConfidenceLevel.HIGH:
                conf_color = "green"
                conf_icon = "✅"
            elif confidence.level == ConfidenceLevel.MEDIUM:
                conf_color = "yellow"
                conf_icon = "⚠️"
            else:
                conf_color = "red"
                conf_icon = "❌"
            
            console.print(f"\n{conf_icon} [bold {conf_color}]Confiance: {confidence.level.value.upper()} ({confidence.overall_score:.2f})[/bold {conf_color}]")
            
            # Facteurs de confiance
            if confidence.factors:
                console.print("\n[dim]Facteurs de confiance:[/dim]")
                for factor, score in confidence.factors.items():
                    console.print(f"  • {factor}: {score:.2f}")
            
            # Risques identifiés
            if confidence.risks:
                console.print(f"\n[yellow]⚠️ Risques identifiés:[/yellow]")
                for risk in confidence.risks:
                    console.print(f"  • {risk}")
        
        # Score de conformité
        overall_score = result.get("overall_conformity_score", 0)
        console.print(f"\n[bold]Score de conformité: {overall_score:.0f}%[/bold]")
        
        # Résultats des clauses (résumé)
        analyses = result.get("conformity_analyses", [])
        if analyses:
            console.print(f"\n[dim]Analysé {len(analyses)} clauses ISO:[/dim]")
            for analysis in analyses[:3]:  # Afficher les 3 premières
                status_icon = "✅" if analysis.conformity_score >= 80 else "⚠️" if analysis.conformity_score >= 50 else "❌"
                console.print(f"  {status_icon} Clause {analysis.clause_number}: {analysis.conformity_score}%")
        
        # Indicateurs de qualité
        report = result.get("final_report")
        if report and "quality_indicators" in report:
            qi = report["quality_indicators"]
            console.print(f"\n[dim]Qualité:[/dim] Récupération={qi['retrieval_quality']}, Preuves={qi['evidence_quality']}")
    
    def _generate_global_report(self, all_results: list):
        """Générer un rapport global pour toutes les sections"""
        
        total_sections = len(all_results)
        
        # Scores agrégés
        total_conformity = sum(r.get("overall_conformity_score", 0) for r in all_results) / total_sections if total_sections > 0 else 0
        
        # Confiance agrégée
        high_conf = sum(1 for r in all_results if r.get("confidence_score") and r["confidence_score"].level == ConfidenceLevel.HIGH)
        medium_conf = sum(1 for r in all_results if r.get("confidence_score") and r["confidence_score"].level == ConfidenceLevel.MEDIUM)
        low_conf = sum(1 for r in all_results if r.get("confidence_score") and r["confidence_score"].level == ConfidenceLevel.LOW)
        
        # Afficher le résumé
        summary_table = Table(title="Résumé de l'analyse globale")
        summary_table.add_column("Métrique", style="cyan")
        summary_table.add_column("Valeur", style="green")
        
        summary_table.add_row("Sections analysées", str(total_sections))
        summary_table.add_row("Score de conformité moyen", f"{total_conformity:.1f}%")
        summary_table.add_row("Sections haute confiance", f"[green]{high_conf}[/green]")
        summary_table.add_row("Sections confiance moyenne", f"[yellow]{medium_conf}[/yellow]")
        summary_table.add_row("Sections faible confiance", f"[red]{low_conf}[/red]")
        
        console.print(summary_table)
        
        # Collecter tous les risques
        all_risks = []
        for result in all_results:
            conf = result.get("confidence_score")
            if conf and conf.risks:
                all_risks.extend(conf.risks)
        
        # Dédupliquer et afficher
        unique_risks = list(set(all_risks))
        if unique_risks:
            console.print("\n[bold yellow]🎯 Actions prioritaires:[/bold yellow]")
            for risk in unique_risks[:5]:  # Top 5
                console.print(f"  • {risk}")
        
        console.print("\n[green]✅ Analyse agentique terminée![/green]")


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="Système RAG ISO 9001/9000 avec Analyse Agentique",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'mode',
        choices=['index', 'chat', 'search', 'stats', 'analyze', 'analyze-agentic'],
        help='Mode d\'exécution'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Forcer la réindexation'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Requête pour le mode search'
    )
    
    parser.add_argument(
        '--document',
        type=str,
        help='Chemin du document à analyser (modes analyze et analyze-agentic)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Nombre de résultats (mode search)'
    )
    
    args = parser.parse_args()
    
    # Initialiser l'application
    app = ISO9001RAGApp()
    
    # Exécuter le mode approprié
    if args.mode == 'index':
        app.index_documents(force_reindex=args.force)
    
    elif args.mode == 'chat':
        app.interactive_mode()
    
    elif args.mode == 'search':
        if not args.query:
            console.print("[red]Erreur: --query requis pour le mode search[/red]")
            sys.exit(1)
        app.search_mode(args.query, k=args.k)
    
    elif args.mode == 'stats':
        app.stats_mode()
    
    elif args.mode == 'analyze':
        if not args.document:
            console.print("[red]Erreur: --document requis pour le mode analyze[/red]")
            console.print("[yellow]Exemple: python main.py analyze --document Doc/formulaire.docx[/yellow]")
            sys.exit(1)
        app.analyze_document_conformity(args.document)
    
    elif args.mode == 'analyze-agentic':
        if not args.document:
            console.print("[red]Erreur: --document requis pour le mode analyze-agentic[/red]")
            console.print("[yellow]Exemple: python main.py analyze-agentic --document Doc/formulaire.docx[/yellow]")
            sys.exit(1)
        app.analyze_document_conformity_agentic(args.document)


if __name__ == "__main__":
    main()