# 🎯 ISO 9001/9000 RAG System with Agentic Conformity Analysis

A powerful Retrieval Augmented Generation (RAG) system specialized in **ISO 9001:2015** and **ISO 9000:2015** quality management standards. Features **intelligent document conformity analysis** using LangGraph agentic workflows, **Groq LLM** (Llama 3.3 70B), and **ChromaDB** vector storage.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-purple.svg)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-orange.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Analysis Modes Comparison](#-analysis-modes-comparison)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

### 🔍 RAG Question-Answering Mode
- **Intelligent indexing** of ISO 9001:2015 and ISO 9000:2015 standards
- **Semantic search** with ChromaDB vector database
- **Contextual answers** with source citations and section references
- **Multilingual support** (French/English) via paraphrase-multilingual-mpnet embeddings
- **Interactive chat** interface with Rich console output

### 📄 Standard Conformity Analysis
- **Document extraction** from PDF and DOCX files
- **Automatic clause mapping** to relevant ISO requirements
- **Conformity scoring** (0-100%) per section
- **Non-conformity detection** with detailed explanations
- **Improvement recommendations** based on ISO requirements

### 🤖 Agentic Analysis (Advanced)
- **LangGraph-powered workflow** with intelligent decision-making
- **Self-healing retrieval** - automatically expands context if insufficient
- **Hallucination detection** - validates claims against source documents
- **Confidence scoring** with risk assessment
- **Iterative refinement** until quality thresholds are met

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ISO 9001 RAG SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Documents   │───▶│  Embeddings  │───▶│    ChromaDB      │  │
│  │  (PDF/DOCX)  │    │  (MPNet)     │    │  Vector Store    │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│         │                                         │             │
│         ▼                                         ▼             │
│  ┌──────────────┐                        ┌──────────────────┐  │
│  │  Document    │                        │  Similarity      │  │
│  │  Extractor   │                        │  Search          │  │
│  └──────────────┘                        └──────────────────┘  │
│         │                                         │             │
│         ▼                                         ▼             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              AGENTIC WORKFLOW (LangGraph)               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
│  │  │Retrieve │─▶│ Grade   │─▶│  Map    │─▶│Analyze  │    │   │
│  │  │Context  │  │Quality  │  │Clauses  │  │Conform. │    │   │
│  │  └─────────┘  └────┬────┘  └─────────┘  └────┬────┘    │   │
│  │       ▲            │                         │          │   │
│  │       │     ┌──────▼──────┐          ┌──────▼──────┐   │   │
│  │       └─────│   Expand    │          │   Check     │   │   │
│  │             │   Context   │          │Hallucinate  │   │   │
│  │             └─────────────┘          └──────┬──────┘   │   │
│  │                                             │          │   │
│  │                                      ┌──────▼──────┐   │   │
│  │                                      │  Generate   │   │   │
│  │                                      │   Report    │   │   │
│  │                                      └─────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│                    ┌──────────────────┐                        │
│                    │   Groq LLM       │                        │
│                    │ (Llama 3.3 70B)  │                        │
│                    └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- [Groq API Key](https://console.groq.com/) (free tier available)

### Step 1: Clone and Setup

```bash
git clone <repository-url>
cd RAG_ISO9001-main
```

### Step 2: Create Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download spaCy French Model (Optional but Recommended)

```bash
python -m spacy download fr_core_news_md
```

---

## ⚙️ Configuration

### 1. Create `.env` File

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at [console.groq.com](https://console.groq.com/).

### 2. Add ISO Documents

Place your ISO standard PDFs in the `data/raw/` folder:

```
data/raw/
├── ISO_9001_2015.pdf
└── ISO_9000_2015.pdf
```

### 3. Configuration Options

Edit `src/config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq LLM model |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `800` | Document chunk size (tokens) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Search results to retrieve |
| `GROQ_TEMPERATURE` | `0.1` | LLM temperature (lower = more focused) |

---

## 📖 Usage

### 1. Index ISO Documents (Required First Time)

```bash
python main.py index
```

This creates the ChromaDB vector database from your ISO PDFs.

To force re-index:
```bash
python main.py index --force
```

### 2. Interactive Chat Mode

```bash
python main.py chat
```

Ask questions about ISO 9001/9000:

```
You: Quelles sont les exigences pour les informations documentées?Assistant: Selon la section 7.5 de l'ISO 9001:2015, les informations documentées doivent...

You: exit
```

### 3. Search Documents

```bash
python main.py search --query "audit interne" --k 5
```

### 4. View Statistics

```bash
python main.py stats
```

### 5. Analyze Document Conformity (Standard)

```bash
python main.py analyze --document "Doc/your_document.docx"
```

### 6. Agentic Analysis (Advanced AI-Powered)

```bash
python main.py analyze-agentic --document "Doc/your_document.docx"
```

This uses the LangGraph workflow for deeper analysis with:
- Automatic context expansion when needed
- Hallucination detection
- Confidence scoring
- Multi-iteration refinement

---

## 🔬 Analysis Modes Comparison

This project offers **two distinct analysis approaches** for evaluating document conformity against ISO 9001/9000. Choose based on your needs:

### Quick Comparison

| Feature | Standard Analysis | Agentic Analysis |
|---------|------------------|------------------|
| **Command** | `analyze` | `analyze-agentic` |
| **Speed** | ⚡ Fast (1-2 min) | 🐢 Slower (3-10 min) |
| **Depth** | Basic conformity check | Deep multi-pass analysis |
| **Context Retrieval** | Single query | Iterative expansion |
| **Quality Assurance** | None | Hallucination detection |
| **Confidence Scoring** | No | Yes (with risk assessment) |
| **Best For** | Quick audits, screening | Detailed reports, certification prep |

---

### 📄 Mode 1: Standard Analysis (`analyze`)

```bash
python main.py analyze --document Doc/your_document.docx
```

#### How It Works

```
Document → Extract Sections → Map to ISO Clauses → Analyze Conformity → Report
    │            │                   │                    │
    └────────────┴───────────────────┴────────────────────┘
                        Single Pass (Linear)
```

#### Process Flow

1. **Document Extraction**: Parses PDF/DOCX into structured sections
2. **Clause Mapping**: Maps each section to relevant ISO 9001 clauses using LLM
3. **Conformity Check**: Evaluates compliance against ISO requirements
4. **Report Generation**: Outputs scores and recommendations

#### Output Example

```
📊 RAPPORT D'ANALYSE DE CONFORMITÉ
═══════════════════════════════════════════════════════════════

📄 Document: formulaire_fabrication.docx
📋 Sections analysées: 15

┌─────────────────────────────────────────────────────────────┐
│ Section: CONTRÔLE QUALITÉ                                   │
├─────────────────────────────────────────────────────────────┤
│ Clauses ISO: 8.6 (Libération des produits)                  │
│ Score: 75% - Partiellement conforme                         │
│                                                              │
│ ✅ Conformités:                                              │
│   • Critères d'acceptation définis                          │
│   • Responsabilités identifiées                             │
│                                                              │
│ ❌ Non-conformités:                                          │
│   • Traçabilité des contrôles insuffisante                  │
│                                                              │
│ 💡 Recommandations:                                          │
│   • Ajouter registre des contrôles effectués                │
└─────────────────────────────────────────────────────────────┘
```

#### When to Use Standard Analysis

✅ **Use when:**
- Quick preliminary assessment needed
- Screening multiple documents
- Time is limited
- Basic conformity overview sufficient

❌ **Avoid when:**
- Preparing for certification audit
- High-stakes compliance decisions
- Need confidence scores
- Complex multi-requirement documents

---

### 🤖 Mode 2: Agentic Analysis (`analyze-agentic`)

```bash
python main.py analyze-agentic --document Doc/your_document.docx
```

#### How It Works

```
                    ┌──────────────────────────────────────────────┐
                    │         AGENTIC WORKFLOW (LangGraph)         │
                    └──────────────────────────────────────────────┘
                                         │
    ┌────────────────────────────────────┼────────────────────────────────────┐
    │                                    ▼                                    │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
    │  │  RETRIEVE   │───▶│   GRADE     │───▶│   EXPAND    │──┐              │
    │  │  ISO Context│    │   Quality   │    │   Context   │  │              │
    │  └─────────────┘    └──────┬──────┘    └─────────────┘  │              │
    │         ▲                  │                   │         │              │
    │         │                  │ Sufficient?       │         │              │
    │         │                  │                   ▼         │              │
    │         │           ┌──────┴──────┐    ┌─────────────┐  │              │
    │         └───────────│   DECIDE    │◀───│   REGRADE   │◀─┘              │
    │          (retry)    └──────┬──────┘    └─────────────┘                 │
    │                            │                                            │
    │                            ▼ Proceed                                    │
    │                    ┌─────────────┐                                      │
    │                    │  MAP ISO    │                                      │
    │                    │  CLAUSES    │                                      │
    │                    └──────┬──────┘                                      │
    │                           │                                             │
    │                           ▼                                             │
    │                    ┌─────────────┐                                      │
    │                    │  ANALYZE    │                                      │
    │                    │ CONFORMITY  │                                      │
    │                    └──────┬──────┘                                      │
    │                           │                                             │
    │                           ▼                                             │
    │                    ┌─────────────┐                                      │
    │                    │   CHECK     │                                      │
    │                    │HALLUCINATE  │───▶ Validates claims against        │
    │                    └──────┬──────┘     source documents                │
    │                           │                                             │
    │                           ▼                                             │
    │                    ┌─────────────┐                                      │
    │                    │ CONFIDENCE  │───▶ Risk assessment                 │
    │                    │   SCORE     │     Reliability rating              │
    │                    └──────┬──────┘                                      │
    │                           │                                             │
    │                           ▼                                             │
    │                    ┌─────────────┐                                      │
    │                    │  GENERATE   │                                      │
    │                    │   REPORT    │                                      │
    │                    └─────────────┘                                      │
    └─────────────────────────────────────────────────────────────────────────┘
```

#### Agentic Nodes Explained

| Node | Purpose | Intelligence |
|------|---------|--------------|
| **Retrieve** | Fetches ISO context from vector DB | Builds optimal queries from section content |
| **Grade** | Assesses retrieval quality | Decides if context is sufficient |
| **Expand** | Gets more context if needed | Up to 3 iterations with refined queries |
| **Map Clauses** | Links sections to ISO requirements | Multi-clause mapping with relevance scores |
| **Analyze** | Evaluates conformity | Detailed evidence-based assessment |
| **Hallucination Check** | Validates LLM claims | Catches unsupported statements |
| **Confidence Score** | Rates reliability | HIGH/MEDIUM/LOW with risk factors |
| **Report** | Generates final output | Structured analysis with recommendations |

#### Output Example

```
============================================================
🔍 AGENTIC ANALYSIS: CONTRÔLE QUALITÉ PRODUIT
============================================================

---RETRIEVE ISO CONTEXT---
Retrieved 10 ISO chunks

---GRADE RETRIEVAL QUALITY---
Quality: GOOD
Missing aspects: traceability requirements

---DECISION: EXPAND CONTEXT---
---EXPAND ISO CONTEXT---
Added 6 chunks (traceability, documentation)

---GRADE RETRIEVAL QUALITY---
Quality: EXCELLENT ✓

---MAP ISO CLAUSES---
Mapped 3 clauses:
  • 8.5.1 - Control of production (relevance: 0.92)
  • 8.5.2 - Identification and traceability (relevance: 0.85)
  • 7.5   - Documented information (relevance: 0.78)

---ANALYZE CONFORMITY---
Conformity Score: 72%

---CHECK HALLUCINATIONS---
✓ No hallucinations detected
All claims supported by ISO source documents

---CALCULATE CONFIDENCE---
Confidence: MEDIUM (0.75)
Factors:
  • Retrieval quality: 0.85
  • Evidence strength: 0.70
  • Clause coverage: 0.72
Risks:
  • Some traceability requirements not fully addressed

============================================================
✅ ANALYSIS COMPLETE
============================================================

┌─────────────────────────────────────────────────────────────┐
│ 📊 FINAL REPORT                                             │
├─────────────────────────────────────────────────────────────┤
│ Overall Score: 72% (Partiellement conforme)                 │
│ Confidence: MEDIUM (75%)                                    │
│ Reliability: Acceptable for internal audit                  │
│                                                              │
│ 🎯 Key Findings:                                            │
│ • Production controls well documented                       │
│ • Traceability gaps in batch identification                 │
│ • Documentation retention period not specified              │
│                                                              │
│ ⚠️ Risks Identified:                                        │
│ • Audit finding likely on clause 8.5.2                      │
│                                                              │
│ 💡 Priority Actions:                                        │
│ 1. Implement batch traceability system                      │
│ 2. Define document retention policy                         │
│ 3. Add verification checkpoints                             │
└─────────────────────────────────────────────────────────────┘
```

#### When to Use Agentic Analysis

✅ **Use when:**
- Preparing for ISO certification audit
- Need confidence scores for decisions
- Complex documents with multiple requirements
- Want hallucination-free analysis
- Detailed reports required

❌ **Avoid when:**
- Quick screening of many documents
- Time-critical situations
- Simple, straightforward documents

---

### 🔄 Running Both for Comparison

For critical documents, run **both** analyses and compare:

```bash
# Standard analysis (fast baseline)
python main.py analyze --document Doc/process_fabrication.docx

# Agentic analysis (deep dive)
python main.py analyze-agentic --document Doc/process_fabrication.docx
```

**Comparison Strategy:**
1. Use **Standard** for initial screening
2. Use **Agentic** for documents flagged as problematic
3. Compare scores - significant differences indicate areas needing attention
4. Trust **Agentic** confidence scores for final decisions

---

## 📁 Project Structure

```
RAG_ISO9001-main/
├── main.py                      # Main CLI application
├── requirements.txt             # Python dependencies
├── .env                         # API keys (create this)
├── README.md                    # This file
│
├── data/
│   ├── raw/                     # ISO PDF documents
│   │   ├── ISO_9001_2015.pdf
│   │   └── ISO_9000_2015.pdf
│   └── processed/               # Processed data cache
│
├── Doc/                         # Company documents to analyze
│   └── *.docx, *.pdf           # Your documents here
│
├── vectorstore/
│   └── chroma_db/              # ChromaDB vector database
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── document_processor.py   # ISO document chunking
│   ├── document_extractor.py   # Company doc extraction (PDF/DOCX)
│   ├── vectorstore_manager.py  # ChromaDB operations
│   ├── rag_system.py           # RAG question-answering
│   ├── conformity_analyzer.py  # Conformity analysis agents
│   ├── iso_agentic_graph.py    # LangGraph workflow
│   ├── iso_agentic_nodes.py    # Agentic workflow nodes
│   ├── iso_agentic_state.py    # Workflow state management
│   └── utils.py                # Utility functions
│
└── notebooks/
    └── test_rag.ipynb          # Jupyter notebook for testing
```

---

## 🔧 How It Works

### RAG Pipeline

1. **Indexing**: ISO documents are split into chunks with metadata (section numbers, titles)
2. **Embedding**: Each chunk is converted to a 768-dim vector using MPNet
3. **Storage**: Vectors are stored in ChromaDB with metadata
4. **Query**: User questions are embedded and matched via cosine similarity
5. **Generation**: Top-K chunks are passed to Groq LLM for answer generation

### Standard vs Agentic Analysis - Technical Deep Dive

#### Standard Analysis Pipeline

```python
# Simplified flow
document = extractor.extract(file_path)           # 1. Extract text
sections = extractor.segment_document(document)    # 2. Segment into sections

for section in sections:
    clauses = clause_mapper.map(section)           # 3. Single LLM call per section
    analysis = conformity_verifier.verify(         # 4. Single verification pass
        section, clauses
    )
    report.add(analysis)                           # 5. Aggregate results
```

**Characteristics:**
- Linear, deterministic flow
- One retrieval per section
- No quality assessment
- Fast but may miss context

#### Agentic Analysis Pipeline

```python
# Simplified flow with decision points
initial_state = initialize_state(section)

while not complete:
    state = retrieve_iso_context(state)            # 1. Retrieve context
    state = grade_retrieval_quality(state)         # 2. AI grades quality
    
    if state.needs_more_context:                   # 3. DECISION POINT
        state = expand_iso_context(state)          #    → Expand if insufficient
        continue                                    #    → Loop back to grade
    
    state = map_iso_clauses(state)                 # 4. Map clauses
    state = analyze_conformity(state)              # 5. Analyze compliance
    state = check_hallucinations(state)            # 6. Validate claims
    
    if state.has_hallucinations:                   # 7. DECISION POINT
        state.retry_count += 1                     #    → Retry if unreliable
        continue
    
    state = calculate_confidence(state)            # 8. Score confidence
    complete = True

return generate_report(state)                      # 9. Final report
```

**Characteristics:**
- Non-linear, adaptive flow
- Self-correcting retrieval (up to 3 expansions)
- Quality gates at each step
- Slower but more thorough

### Conformity Scoring

| Score | Status | Description |
|-------|--------|-------------|
| 80-100% | ✅ Conforme | All major requirements met |
| 50-79% | ⚠️ Partiellement conforme | Some requirements missing |
| 0-49% | ❌ Non conforme | Major gaps identified |

### Confidence Levels (Agentic Only)

| Level | Score | Meaning |
|-------|-------|---------|
| **HIGH** | > 80% | Analysis highly reliable, suitable for external audit |
| **MEDIUM** | 50-80% | Analysis acceptable, review flagged items |
| **LOW** | < 50% | Analysis uncertain, manual review recommended |

---

## 🐛 Troubleshooting

### Common Issues

**1. "GROQ_API_KEY n'est pas définie"**
```bash
# Create .env file with your API key
echo "GROQ_API_KEY=your_key_here" > .env
```

**2. "Base de données ISO non indexée"**
```bash
python main.py index
```

**3. "Fichier introuvable" for document analysis**
```bash
# Use path relative to project root
python main.py analyze --document "Doc/filename.docx"
```

**4. Recursion limit error in agentic analysis**

This has been fixed. If you still encounter it, the system will automatically stop after 3 context expansion attempts.

**5. ChromaDB telemetry errors**

These are harmless warnings. ChromaDB telemetry is disabled but may still show errors.

**6. spaCy model not found**
```bash
python -m spacy download fr_core_news_md
```

### Performance Tips

- **First run is slow**: Embedding model downloads (~500MB)
- **Use SSD storage**: ChromaDB benefits from fast disk I/O
- **GPU acceleration**: Set `device: 'cuda'` in `vectorstore_manager.py` if available

---

## 📚 Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | LLM orchestration framework |
| `langchain-groq` | Groq LLM integration |
| `langgraph` | Agentic workflow graphs |
| `chromadb` | Vector database |
| `sentence-transformers` | Embedding models |
| `pypdf` / `pdfplumber` | PDF extraction |
| `python-docx` | Word document extraction |
| `spacy` | NLP for French text |
| `rich` | Beautiful terminal output |
| `pydantic` | Data validation |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is provided as-is for educational and professional use.

---

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for fast LLM inference
- [LangChain](https://langchain.com/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- ISO for the quality management standards

---

**Built with ❤️ for quality management professionals**