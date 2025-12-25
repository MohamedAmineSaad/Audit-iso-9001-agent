"""
ISO 9001 Agentic Workflow Nodes
Intelligent nodes for ISO conformity analysis using LangGraph
"""
import logging
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import json

logger = logging.getLogger(__name__)


# === Node 1: Retrieve ISO Context ===

def retrieve_iso_context(state: Dict[str, Any], vectorstore_manager, rag_system) -> Dict[str, Any]:
    """
    Retrieve relevant ISO clauses from ChromaDB
    
    Uses both semantic search and keyword matching for comprehensive retrieval
    """
    print("---RETRIEVE ISO CONTEXT---")
    
    section = state["document_section"]
    
    # Build retrieval query
    query = f"""
    Document section: {section.title}
    Type: {section.section_type}
    Keywords: {', '.join(section.keywords)}
    Content excerpt: {section.content[:500]}
    
    Find relevant ISO 9001 and ISO 9000 clauses.
    """
    
    try:
        # Retrieve from vectorstore
        results = vectorstore_manager.similarity_search_with_score(
            query=query,
            k=10  # Cast wider net initially
        )
        
        # Format results
        retrieved_chunks = []
        for doc, score in results:
            retrieved_chunks.append({
                "content": doc.page_content,
                "clause": doc.metadata.get("section_number", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "similarity": float(1 - score),
                "metadata": doc.metadata
            })
        
        state["retrieved_iso_chunks"] = retrieved_chunks
        state["workflow_stage"] = "context_retrieved"
        state["audit_notes"].append(f"Retrieved {len(retrieved_chunks)} ISO chunks")
        
        return state
        
    except Exception as e:
        logger.error(f"Error retrieving ISO context: {e}")
        state["errors"].append(f"Retrieval error: {str(e)}")
        return state


# === Node 2: Grade Retrieval Quality ===

def grade_retrieval_quality(state: Dict[str, Any], llm: ChatGroq) -> Dict[str, Any]:
    """
    Assess whether retrieved ISO context is sufficient for conformity analysis
    
    This is the key agentic decision point - determines if we need more context
    """
    print("---GRADE RETRIEVAL QUALITY---")
    
    section = state["document_section"]
    chunks = state["retrieved_iso_chunks"]
    
    if not chunks:
        from iso_agentic_state import RetrievalQuality, RetrievalQualityAssessment
        state["retrieval_quality"] = RetrievalQualityAssessment(
            quality=RetrievalQuality.INSUFFICIENT,
            reason="No ISO context retrieved",
            missing_aspects=["All aspects"],
            needs_expansion=True,
            suggested_queries=["ISO 9001 general requirements"]
        )
        state["needs_more_context"] = True
        return state
    
    # Prepare grading prompt
    grading_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an ISO 9001 expert assessing retrieval quality.

Evaluate if the retrieved ISO context is SUFFICIENT to assess conformity for this document section.

Consider:
1. Are relevant clause requirements present?
2. Is the context detailed enough to identify non-conformities?
3. Are related clauses (traceability, documentation) covered?

Respond ONLY with JSON (no markdown):
{{
  "quality": "excellent|good|weak|insufficient",
  "reason": "Brief explanation",
  "missing_aspects": ["aspect1", "aspect2"],
  "needs_expansion": true/false,
  "suggested_queries": ["query1 if expansion needed"]
}}"""),
        ("human", """DOCUMENT SECTION:
Title: {title}
Type: {section_type}
Content: {content}

RETRIEVED ISO CONTEXT:
{iso_context}

Assess quality:""")
    ])
    
    # Build context summary
    iso_context = "\n\n".join([
        f"Clause {c['clause']}: {c['content'][:300]}..."
        for c in chunks[:5]
    ])
    
    try:
        prompt = grading_prompt.format(
            title=section.title,
            section_type=section.section_type or "general",
            content=section.content[:800],
            iso_context=iso_context
        )
        
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Clean JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        # Create assessment
        from iso_agentic_state import RetrievalQuality, RetrievalQualityAssessment
        
        quality_map = {
            "excellent": RetrievalQuality.EXCELLENT,
            "good": RetrievalQuality.GOOD,
            "weak": RetrievalQuality.WEAK,
            "insufficient": RetrievalQuality.INSUFFICIENT
        }
        
        assessment = RetrievalQualityAssessment(
            quality=quality_map[result["quality"]],
            reason=result["reason"],
            missing_aspects=result.get("missing_aspects", []),
            needs_expansion=result.get("needs_expansion", False),
            suggested_queries=result.get("suggested_queries", [])
        )
        
        state["retrieval_quality"] = assessment
        state["needs_more_context"] = assessment.needs_expansion
        state["workflow_stage"] = "retrieval_graded"
        state["audit_notes"].append(f"Retrieval quality: {assessment.quality.value}")
        
        print(f"---QUALITY: {assessment.quality.value}---")
        if assessment.needs_expansion:
            print(f"---NEEDS EXPANSION: {assessment.missing_aspects}---")
        
        return state
        
    except Exception as e:
        logger.error(f"Error grading retrieval: {e}")
        # Default to weak quality
        from iso_agentic_state import RetrievalQuality, RetrievalQualityAssessment
        state["retrieval_quality"] = RetrievalQualityAssessment(
            quality=RetrievalQuality.WEAK,
            reason=f"Grading error: {str(e)}",
            missing_aspects=[],
            needs_expansion=False,
            suggested_queries=[]
        )
        state["needs_more_context"] = False
        return state


# === Node 3: Expand Context (Conditional) ===

def expand_iso_context(state: Dict[str, Any], vectorstore_manager, rag_system) -> Dict[str, Any]:
    """
    Expand retrieval with more targeted queries
    
    Only called when retrieval quality is weak/insufficient
    """
    print("---EXPAND ISO CONTEXT---")
    
    assessment = state["retrieval_quality"]
    state["expansion_attempts"] += 1
    
    # Prevent infinite loops
    if state["expansion_attempts"] > 3:
        print("---MAX EXPANSIONS REACHED---")
        state["needs_more_context"] = False
        state["audit_notes"].append("Warning: Max context expansion reached")
        return state
    
    # Use suggested queries or fallback
    queries = assessment.suggested_queries or [
        f"ISO 9001 {aspect}" for aspect in assessment.missing_aspects[:2]
    ]
    
    expanded_results = []
    
    for query in queries:
        try:
            results = vectorstore_manager.similarity_search_with_score(
                query=query,
                k=3
            )
            
            for doc, score in results:
                expanded_results.append({
                    "content": doc.page_content,
                    "clause": doc.metadata.get("section_number", "unknown"),
                    "source": doc.metadata.get("source", "unknown"),
                    "similarity": float(1 - score),
                    "metadata": doc.metadata,
                    "expansion_query": query
                })
        except Exception as e:
            logger.warning(f"Expansion query failed: {e}")
    
    # Merge with existing
    state["expanded_context"] = expanded_results
    state["retrieved_iso_chunks"].extend(expanded_results)
    
    # Deduplicate by clause
    seen_clauses = set()
    unique_chunks = []
    for chunk in state["retrieved_iso_chunks"]:
        clause = chunk["clause"]
        if clause not in seen_clauses:
            seen_clauses.add(clause)
            unique_chunks.append(chunk)
    
    state["retrieved_iso_chunks"] = unique_chunks
    state["workflow_stage"] = "context_expanded"
    state["audit_notes"].append(f"Expanded context: +{len(expanded_results)} chunks")
    
    print(f"---ADDED {len(expanded_results)} CHUNKS---")
    
    return state


# === Node 4: Map ISO Clauses (Your Existing Logic) ===

def map_iso_clauses(state: Dict[str, Any], clause_mapper) -> Dict[str, Any]:
    """
    Map document section to specific ISO clauses
    
    Uses your existing ClauseMappingAgent but with validated context
    """
    print("---MAP ISO CLAUSES---")
    
    section = state["document_section"]
    
    try:
        # Use your existing mapper
        mappings = clause_mapper.map_section_to_clauses(section, max_clauses=5)
        
        # Calculate mapping confidence
        if mappings:
            avg_relevance = sum(m.relevance_score for m in mappings) / len(mappings)
            state["mapping_confidence"] = avg_relevance
        else:
            state["mapping_confidence"] = 0.0
        
        state["mapped_clauses"] = mappings
        state["workflow_stage"] = "clauses_mapped"
        state["audit_notes"].append(f"Mapped {len(mappings)} clauses")
        
        print(f"---MAPPED {len(mappings)} CLAUSES---")
        
        return state
        
    except Exception as e:
        logger.error(f"Clause mapping error: {e}")
        state["errors"].append(f"Mapping error: {str(e)}")
        state["mapped_clauses"] = []
        return state


# === Node 5: Analyze Conformity (Your Existing Logic) ===

def analyze_conformity(state: Dict[str, Any], conformity_verifier) -> Dict[str, Any]:
    """
    Analyze conformity using your existing ConformityVerificationAgent
    
    Now with validated context and explicit clauses
    """
    print("---ANALYZE CONFORMITY---")
    
    section = state["document_section"]
    mappings = state["mapped_clauses"]
    
    if not mappings:
        print("---NO CLAUSES TO ANALYZE---")
        state["conformity_analyses"] = []
        state["evidence_quality"] = "weak"
        return state
    
    try:
        # Use your existing verifier
        analyses = conformity_verifier.verify_multiple_clauses(section, mappings)
        
        state["conformity_analyses"] = analyses
        
        # Assess evidence quality
        if analyses:
            avg_score = sum(a.conformity_score for a in analyses) / len(analyses)
            if avg_score >= 75:
                state["evidence_quality"] = "strong"
            elif avg_score >= 50:
                state["evidence_quality"] = "moderate"
            else:
                state["evidence_quality"] = "weak"
            
            state["overall_conformity_score"] = avg_score
        
        state["workflow_stage"] = "conformity_analyzed"
        state["audit_notes"].append(f"Analyzed {len(analyses)} clauses")
        
        print(f"---CONFORMITY SCORE: {state['overall_conformity_score']:.0f}%---")
        
        return state
        
    except Exception as e:
        logger.error(f"Conformity analysis error: {e}")
        state["errors"].append(f"Analysis error: {str(e)}")
        state["conformity_analyses"] = []
        return state


# === Node 6: Check for Hallucinations ===

def check_hallucinations(state: Dict[str, Any], llm: ChatGroq) -> Dict[str, Any]:
    """
    Verify that all conformity claims are backed by evidence
    
    Adapted from Agentic RAG hallucination grader
    """
    print("---CHECK HALLUCINATIONS---")
    
    section = state["document_section"]
    analyses = state["conformity_analyses"]
    
    if not analyses:
        state["hallucination_check"] = None
        return state
    
    # Check each analysis
    unsupported_claims = []
    evidence_gaps = []
    
    for analysis in analyses:
        # Check if conformity elements have corresponding evidence
        if analysis.conformity_elements and not analysis.evidence:
            unsupported_claims.append(
                f"Clause {analysis.clause_number}: Claims conformity without extracted evidence"
            )
        
        # Check if non-conformities are substantiated
        if analysis.non_conformities:
            # Should have concrete quotes from document
            for nc in analysis.non_conformities:
                if "missing" in nc.lower() or "absence" in nc.lower():
                    # These are valid (absence of evidence)
                    continue
                else:
                    # Should have evidence
                    evidence_gaps.append(f"Clause {analysis.clause_number}: {nc}")
    
    has_hallucination = len(unsupported_claims) > 0
    penalty = min(len(unsupported_claims) * 0.15, 0.5)  # Cap at 50% penalty
    
    from iso_agentic_state import HallucinationCheck
    
    check = HallucinationCheck(
        has_hallucination=has_hallucination,
        unsupported_claims=unsupported_claims,
        evidence_gaps=evidence_gaps,
        confidence_penalty=penalty
    )
    
    state["hallucination_check"] = check
    state["workflow_stage"] = "hallucination_checked"
    
    if has_hallucination:
        print(f"---HALLUCINATION DETECTED: {len(unsupported_claims)} claims---")
        state["audit_notes"].append(f"Warning: {len(unsupported_claims)} unsupported claims")
    else:
        print("---NO HALLUCINATIONS---")
    
    return state


# === Node 7: Calculate Final Confidence ===

def calculate_confidence(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive confidence score
    
    This is what makes the agent "consultant-grade"
    """
    print("---CALCULATE CONFIDENCE---")
    
    from iso_agentic_state import calculate_confidence_score
    
    confidence = calculate_confidence_score(state)
    
    state["confidence_score"] = confidence
    state["workflow_stage"] = "confidence_calculated"
    
    print(f"---CONFIDENCE: {confidence.level.value.upper()} ({confidence.overall_score:.2f})---")
    if confidence.risks:
        print(f"---RISKS: {len(confidence.risks)}---")
    
    return state


# === Node 8: Generate Final Report ===

def generate_final_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate audit-style final report
    """
    print("---GENERATE FINAL REPORT---")
    
    section = state["document_section"]
    analyses = state["conformity_analyses"]
    confidence = state["confidence_score"]
    
    report = {
        "section_title": section.title,
        "section_type": section.section_type,
        "timestamp": str(__import__("datetime").datetime.now()),
        
        "conformity_score": state["overall_conformity_score"],
        "confidence": {
            "score": confidence.overall_score if confidence else 0,
            "level": confidence.level.value if confidence else "unknown",
            "risks": confidence.risks if confidence else [],
            "reliability": confidence.reliability_notes if confidence else ""
        },
        
        "clause_analyses": [
            {
                "clause": a.clause_number,
                "title": a.clause_title,
                "status": a.status.value,
                "score": a.conformity_score,
                "conformity_elements": a.conformity_elements,
                "non_conformities": a.non_conformities,
                "recommendations": a.recommendations,
                "evidence": a.evidence
            }
            for a in analyses
        ],
        
        "quality_indicators": {
            "retrieval_quality": state["retrieval_quality"].quality.value if state["retrieval_quality"] else "unknown",
            "evidence_quality": state["evidence_quality"],
            "hallucination_check": "passed" if not (state["hallucination_check"] and state["hallucination_check"].has_hallucination) else "warning"
        },
        
        "audit_trail": state["audit_notes"],
        "errors": state["errors"]
    }
    
    state["final_report"] = report
    state["workflow_stage"] = "completed"
    
    print("---REPORT GENERATED---")
    
    return state
