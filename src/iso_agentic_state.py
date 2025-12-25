"""
ISO 9001 Agentic Analysis State
State management for the agentic ISO auditor workflow
"""
from typing import TypedDict, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Reuse your existing types
from src.document_extractor import DocumentSection
from src.conformity_analyzer import ISOClauseMapping, ConformityAnalysis


class RetrievalQuality(str, Enum):
    """Quality assessment of retrieved ISO context"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"


class ConfidenceLevel(str, Enum):
    """Confidence in the analysis result"""
    HIGH = "high"       # > 80%
    MEDIUM = "medium"   # 50-80%
    LOW = "low"         # < 50%


@dataclass
class RetrievalQualityAssessment:
    """Assessment of whether retrieved ISO context is sufficient"""
    quality: RetrievalQuality
    reason: str
    missing_aspects: List[str]
    needs_expansion: bool
    suggested_queries: List[str]  # For refined retrieval


@dataclass
class HallucinationCheck:
    """Result of hallucination detection"""
    has_hallucination: bool
    unsupported_claims: List[str]
    evidence_gaps: List[str]
    confidence_penalty: float  # 0.0 to 1.0


@dataclass
class ConfidenceScore:
    """Comprehensive confidence scoring"""
    overall_score: float  # 0.0 to 1.0
    level: ConfidenceLevel
    factors: Dict[str, float]  # breakdown by factor
    risks: List[str]
    reliability_notes: str


class ISOAnalysisState(TypedDict):
    """
    State for agentic ISO 9001 conformity analysis workflow
    
    This state flows through the LangGraph nodes and accumulates
    information at each step.
    """
    
    # === Input ===
    document_section: DocumentSection
    analysis_depth: str  # "quick" | "standard" | "detailed"
    
    # === ISO Context Retrieval ===
    retrieved_iso_chunks: List[Dict]  # Raw ChromaDB results
    retrieval_quality: Optional[RetrievalQualityAssessment]
    
    # === Clause Mapping ===
    candidate_clauses: List[ISOClauseMapping]
    mapped_clauses: List[ISOClauseMapping]  # Validated/filtered
    mapping_confidence: float
    
    # === Context Expansion (if needed) ===
    needs_more_context: bool
    expanded_context: Optional[List[Dict]]
    expansion_attempts: int  # Prevent infinite loops
    
    # === Conformity Analysis ===
    conformity_analyses: List[ConformityAnalysis]
    evidence_quality: str  # "strong" | "moderate" | "weak"
    
    # === Quality Checks ===
    hallucination_check: Optional[HallucinationCheck]
    evidence_verification: Dict[str, bool]  # clause -> has_evidence
    
    # === Final Scoring ===
    confidence_score: Optional[ConfidenceScore]
    overall_conformity_score: float
    
    # === Control Flow ===
    workflow_stage: str  # Track where we are in the graph
    retry_count: int
    errors: List[str]
    
    # === Output ===
    final_report: Optional[Dict]
    audit_notes: List[str]


# === Helper Functions for State Management ===

def initialize_analysis_state(section: DocumentSection) -> ISOAnalysisState:
    """Initialize a new analysis state"""
    return ISOAnalysisState(
        document_section=section,
        analysis_depth="standard",
        retrieved_iso_chunks=[],
        retrieval_quality=None,
        candidate_clauses=[],
        mapped_clauses=[],
        mapping_confidence=0.0,
        needs_more_context=False,
        expanded_context=None,
        expansion_attempts=0,
        conformity_analyses=[],
        evidence_quality="unknown",
        hallucination_check=None,
        evidence_verification={},
        confidence_score=None,
        overall_conformity_score=0.0,
        workflow_stage="initialized",
        retry_count=0,
        errors=[],
        final_report=None,
        audit_notes=[]
    )


def update_workflow_stage(state: ISOAnalysisState, stage: str) -> ISOAnalysisState:
    """Update the workflow stage and add audit note"""
    state["workflow_stage"] = stage
    state["audit_notes"].append(f"Stage: {stage}")
    return state


def add_error(state: ISOAnalysisState, error: str) -> ISOAnalysisState:
    """Add an error to the state"""
    state["errors"].append(error)
    state["audit_notes"].append(f"Error: {error}")
    return state


def calculate_confidence_score(state: ISOAnalysisState) -> ConfidenceScore:
    """
    Calculate comprehensive confidence score based on multiple factors
    
    Factors:
    - Retrieval quality (30%)
    - Mapping confidence (20%)
    - Evidence quality (25%)
    - Hallucination check (25%)
    """
    factors = {}
    
    # Retrieval quality
    if state["retrieval_quality"]:
        quality_map = {
            RetrievalQuality.EXCELLENT: 1.0,
            RetrievalQuality.GOOD: 0.8,
            RetrievalQuality.WEAK: 0.5,
            RetrievalQuality.INSUFFICIENT: 0.2
        }
        factors["retrieval_quality"] = quality_map[state["retrieval_quality"].quality] * 0.3
    else:
        factors["retrieval_quality"] = 0.0
    
    # Mapping confidence
    factors["mapping_confidence"] = state["mapping_confidence"] * 0.2
    
    # Evidence quality
    evidence_map = {"strong": 1.0, "moderate": 0.7, "weak": 0.4, "unknown": 0.0}
    factors["evidence_quality"] = evidence_map[state["evidence_quality"]] * 0.25
    
    # Hallucination penalty
    if state["hallucination_check"]:
        hallucination_factor = 1.0 - state["hallucination_check"].confidence_penalty
        factors["hallucination_check"] = hallucination_factor * 0.25
    else:
        factors["hallucination_check"] = 0.25
    
    # Calculate overall
    overall = sum(factors.values())
    
    # Determine level
    if overall >= 0.8:
        level = ConfidenceLevel.HIGH
    elif overall >= 0.5:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW
    
    # Identify risks
    risks = []
    if state["retrieval_quality"] and state["retrieval_quality"].quality in [
        RetrievalQuality.WEAK, RetrievalQuality.INSUFFICIENT
    ]:
        risks.append("Limited ISO context available")
    
    if state["hallucination_check"] and state["hallucination_check"].has_hallucination:
        risks.append("Potential unsupported claims detected")
    
    if state["evidence_quality"] == "weak":
        risks.append("Weak evidence from document")
    
    if state["expansion_attempts"] > 2:
        risks.append("Required multiple context expansions")
    
    # Reliability notes
    notes = f"Analysis based on {len(state['mapped_clauses'])} ISO clauses. "
    if state["retrieval_quality"]:
        notes += f"Context quality: {state['retrieval_quality'].quality.value}."
    
    return ConfidenceScore(
        overall_score=overall,
        level=level,
        factors=factors,
        risks=risks,
        reliability_notes=notes
    )
