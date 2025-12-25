"""
ISO 9001 Agentic Analysis Graph
Intelligent workflow orchestration using LangGraph
"""
from langgraph.graph import StateGraph, END
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Import your existing components
from src.vectorstore_manager import ChromaDBManager
from src.rag_system import ISO9001RAGSystem
from src.conformity_analyzer import ClauseMappingAgent, ConformityVerificationAgent

# Import the new agentic nodes
from iso_agentic_nodes import (
    retrieve_iso_context,
    grade_retrieval_quality,
    expand_iso_context,
    map_iso_clauses,
    analyze_conformity,
    check_hallucinations,
    calculate_confidence,
    generate_final_report
)

from iso_agentic_state import ISOAnalysisState


# === Decision Functions (Conditional Edges) ===

def decide_if_context_expansion_needed(state: Dict[str, Any]) -> str:
    """
    Decide if we need to expand ISO context retrieval
    
    This is a KEY agentic decision point
    """
    print("---DECIDE: EXPANSION NEEDED?---")
    
    needs_more = state.get("needs_more_context", False)
    attempts = state.get("expansion_attempts", 0)
    
    # Hard limit to prevent infinite loops
    if attempts >= 3:
        print("---DECISION: PROCEED (max expansions reached)---")
        return "proceed"
    
    if needs_more:
        print("---DECISION: EXPAND CONTEXT---")
        return "expand"
    else:
        print("---DECISION: PROCEED (context sufficient)---")
        return "proceed"


def decide_if_regrade_needed(state: Dict[str, Any]) -> str:
    """
    After expansion, decide if we should regrade retrieval quality
    """
    print("---DECIDE: REGRADE?---")
    
    expanded = state.get("expanded_context")
    attempts = state.get("expansion_attempts", 0)
    
    # Don't loop back if we've hit max expansions
    if attempts >= 3:
        print("---DECISION: PROCEED (max expansions, skip regrade)---")
        return "proceed"
    
    if expanded and len(expanded) > 0:
        print("---DECISION: REGRADE WITH NEW CONTEXT---")
        return "regrade"
    else:
        print("---DECISION: SKIP REGRADE---")
        return "proceed"


def decide_if_analysis_reliable(state: Dict[str, Any]) -> str:
    """
    Decide if analysis is reliable enough or needs retry
    
    Based on confidence score and hallucination check
    """
    print("---DECIDE: ANALYSIS RELIABLE?---")
    
    confidence = state.get("confidence_score")
    hallucination = state.get("hallucination_check")
    retry_count = state.get("retry_count", 0)
    
    # Check for critical failures
    if retry_count >= 2:
        print("---DECISION: ACCEPT (max retries reached)---")
        return "accept"
    
    # Check confidence
    if confidence:
        if confidence.overall_score < 0.3:
            print(f"---DECISION: RETRY (low confidence: {confidence.overall_score:.2f})---")
            state["retry_count"] = retry_count + 1
            return "retry"
    
    # Check hallucinations
    if hallucination and hallucination.has_hallucination:
        if len(hallucination.unsupported_claims) > 3:
            print("---DECISION: RETRY (too many hallucinations)---")
            state["retry_count"] = retry_count + 1
            return "retry"
    
    print("---DECISION: ACCEPT---")
    return "accept"


# === Build the Graph ===

class ISOAgenticAnalyzer:
    """
    Agentic ISO 9001 Conformity Analyzer
    
    Combines your existing ISO domain logic with intelligent decision-making
    """
    
    def __init__(self, 
                 vectorstore_manager: ChromaDBManager,
                 rag_system: ISO9001RAGSystem):
        """
        Initialize the agentic analyzer
        
        Args:
            vectorstore_manager: Your existing ChromaDB manager
            rag_system: Your existing RAG system
        """
        self.vectorstore_manager = vectorstore_manager
        self.rag_system = rag_system
        self.llm = rag_system.llm
        
        # Initialize your existing agents
        self.clause_mapper = ClauseMappingAgent(
            llm=self.llm,
            rag_system=self.rag_system
        )
        
        self.conformity_verifier = ConformityVerificationAgent(
            llm=self.llm,
            rag_system=self.rag_system
        )
        
        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow
        
        This is the "brain" of the agentic system
        """
        
        # Create workflow
        workflow = StateGraph(ISOAnalysisState)
        
        # === Add Nodes ===
        
        workflow.add_node(
            "retrieve",
            lambda s: retrieve_iso_context(s, self.vectorstore_manager, self.rag_system)
        )
        
        workflow.add_node(
            "grade_retrieval",
            lambda s: grade_retrieval_quality(s, self.llm)
        )
        
        workflow.add_node(
            "expand_context",
            lambda s: expand_iso_context(s, self.vectorstore_manager, self.rag_system)
        )
        
        workflow.add_node(
            "map_clauses",
            lambda s: map_iso_clauses(s, self.clause_mapper)
        )
        
        workflow.add_node(
            "analyze_conformity",
            lambda s: analyze_conformity(s, self.conformity_verifier)
        )
        
        workflow.add_node(
            "check_hallucinations",
            lambda s: check_hallucinations(s, self.llm)
        )
        
        workflow.add_node(
            "calculate_confidence",
            calculate_confidence
        )
        
        workflow.add_node(
            "generate_report",
            generate_final_report
        )
        
        # === Define Flow ===
        
        # Start with retrieval
        workflow.set_entry_point("retrieve")
        
        # Retrieve → Grade
        workflow.add_edge("retrieve", "grade_retrieval")
        
        # Grade → Expand or Map (conditional)
        workflow.add_conditional_edges(
            "grade_retrieval",
            decide_if_context_expansion_needed,
            {
                "expand": "expand_context",
                "proceed": "map_clauses"
            }
        )
        
        # Expand → Regrade or Map (conditional)
        workflow.add_conditional_edges(
            "expand_context",
            decide_if_regrade_needed,
            {
                "regrade": "grade_retrieval",  # Loop back to regrade
                "proceed": "map_clauses"
            }
        )
        
        # Map → Analyze
        workflow.add_edge("map_clauses", "analyze_conformity")
        
        # Analyze → Check Hallucinations
        workflow.add_edge("analyze_conformity", "check_hallucinations")
        
        # Check → Calculate Confidence
        workflow.add_edge("check_hallucinations", "calculate_confidence")
        
        # Confidence → Retry or Report (conditional)
        workflow.add_conditional_edges(
            "calculate_confidence",
            decide_if_analysis_reliable,
            {
                "retry": "retrieve",  # Start over with retry
                "accept": "generate_report"
            }
        )
        
        # Report → END
        workflow.add_edge("generate_report", END)
        
        return workflow
    
    def analyze_section(self, section) -> Dict[str, Any]:
        """
        Analyze a document section for ISO conformity
        
        This is your main entry point
        
        Args:
            section: DocumentSection from your extractor
            
        Returns:
            Final analysis report with confidence scores
        """
        from iso_agentic_state import initialize_analysis_state
        
        # Initialize state
        initial_state = initialize_analysis_state(section)
        
        print(f"\n{'='*60}")
        print(f"🔍 AGENTIC ANALYSIS: {section.title}")
        print(f"{'='*60}\n")
        
        # Run the graph with increased recursion limit
        final_state = self.app.invoke(
            initial_state,
            {"recursion_limit": 100}
        )
        
        print(f"\n{'='*60}")
        print("✅ ANALYSIS COMPLETE")
        print(f"{'='*60}\n")
        
        return final_state
    
    def export_graph_visualization(self, output_path: str = "iso_agentic_graph.png"):
        """Export graph visualization"""
        try:
            self.app.get_graph().draw_mermaid_png(output_file_path=output_path)
            print(f"Graph visualization saved to: {output_path}")
        except Exception as e:
            logger.warning(f"Could not export graph: {e}")


# === Usage Example ===

def create_agentic_analyzer(groq_api_key: str,
                           vectorstore_dir: str,
                           collection_name: str,
                           embedding_model: str) -> ISOAgenticAnalyzer:
    """
    Factory function to create the agentic analyzer
    
    Args:
        groq_api_key: Your Groq API key
        vectorstore_dir: Path to ChromaDB
        collection_name: Collection name
        embedding_model: Embedding model name
        
    Returns:
        Configured ISOAgenticAnalyzer
    """
    
    # Initialize your existing components
    vectorstore_manager = ChromaDBManager(
        persist_directory=vectorstore_dir,
        collection_name=collection_name,
        embedding_model=embedding_model
    )
    
    rag_system = ISO9001RAGSystem(
        groq_api_key=groq_api_key,
        vectorstore_manager=vectorstore_manager,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )
    
    # Create agentic analyzer
    analyzer = ISOAgenticAnalyzer(
        vectorstore_manager=vectorstore_manager,
        rag_system=rag_system
    )
    
    return analyzer


if __name__ == "__main__":
    # Example: Test the graph structure
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Create analyzer
    analyzer = create_agentic_analyzer(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        vectorstore_dir=Path("vectorstore/chroma_db"),
        collection_name="iso_9001_knowledge",
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    
    # Export graph visualization
    analyzer.export_graph_visualization()
    
    print("✅ Agentic ISO Analyzer initialized successfully")
    print("Graph structure exported to: iso_agentic_graph.png")
