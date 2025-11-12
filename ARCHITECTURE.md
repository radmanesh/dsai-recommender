# System Architecture

## Overview

The Agentic Research Matchmaker is a multi-agent system that uses semantic search and LLM-based reasoning to match PhD proposals with relevant faculty members.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│  (CLI / API / Future: Web UI via GADK)                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Orchestrator Agent                            │
│  • Coordinates workflow                                         │
│  • Manages context                                              │
│  • Handles multi-proposal comparison                            │
└──────┬──────────────────┬──────────────────┬───────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
│  Proposal   │  │   Faculty    │  │ Recommendation   │
│  Analyzer   │  │  Retrieval   │  │     Agent        │
│   Agent     │  │    Agent     │  │                  │
└─────────────┘  └──────────────┘  └──────────────────┘
       │                  │                  │
       │                  ▼                  │
       │         ┌─────────────────┐        │
       │         │ Index Manager   │        │
       │         │  (LlamaIndex)   │        │
       │         └────────┬────────┘        │
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Model Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Qwen2.5-Coder│  │ BGE Embeddings│  │  ChromaDB    │         │
│  │ (via HF API) │  │  (HuggingFace)│  │ Vector Store │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Sources                               │
│  • DSAI-Faculties.csv (structured)                             │
│  • PDF documents (unstructured)                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Ingestion Pipeline

**Purpose**: Load, process, and index faculty data from multiple sources

**Components**:
- **CSV Loader** (`src/ingestion/csv_loader.py`)
  - Parses faculty metadata from CSV
  - Creates LlamaIndex Document objects
  - Enriches with structured metadata

- **PDF Loader** (`src/ingestion/pdf_loader.py`)
  - Extracts text from PDF documents
  - Infers document type (CV, paper, proposal)
  - Attempts to extract faculty names from filenames

- **Ingestion Pipeline** (`src/ingestion/pipeline.py`)
  - Combines documents from all sources
  - Chunks text using SentenceSplitter
  - Generates embeddings
  - Stores in ChromaDB

**Flow**:
```
CSV + PDFs → Documents → Chunking → Embedding → ChromaDB
```

### 2. Indexing & Retrieval Layer

**Purpose**: Manage vector storage and semantic search

**Components**:
- **Vector Store** (`src/indexing/vector_store.py`)
  - ChromaDB client management
  - Collection creation and deletion
  - Statistics and monitoring

- **Index Builder** (`src/indexing/index_builder.py`)
  - Creates VectorStoreIndex from ChromaDB
  - Provides query engines with different modes
  - Manages retrievers for similarity search

**Retrieval Modes**:
1. **Similarity Search**: Returns top-k most similar documents
2. **Query with LLM**: Synthesizes answer using retrieved context
3. **Custom Retrieval**: Flexible retrieval with custom parameters

### 3. Agent System

#### 3.1 Proposal Analyzer Agent

**Purpose**: Extract structured information from proposals

**Input**: PDF file or text
**Output**: ProposalAnalysis object

```python
ProposalAnalysis:
  - topics: List[str]
  - methods: List[str]
  - domain: str
  - application_areas: List[str]
  - key_phrases: List[str]
  - summary: str
```

**Process**:
1. Load and parse proposal (PDF → text)
2. Create analysis prompt for LLM
3. Use Qwen2.5-Coder to extract structured info
4. Parse LLM response into structured format

#### 3.2 Faculty Retrieval Agent

**Purpose**: Find relevant faculty using semantic search

**Input**: ProposalAnalysis or query string
**Output**: List of FacultyMatch objects

```python
FacultyMatch:
  - score: float
  - faculty_name: Optional[str]
  - text: str
  - metadata: Dict
  - source_type: str
```

**Process**:
1. Convert analysis to search query
2. Retrieve top-k similar documents from ChromaDB
3. Group results by faculty member
4. Rank by similarity score

#### 3.3 Recommendation Agent

**Purpose**: Generate explanations and rank faculty

**Input**: Faculty matches + proposal analysis
**Output**: List of FacultyRecommendation objects

```python
FacultyRecommendation:
  - rank: int
  - faculty_name: str
  - score: float
  - explanation: str
  - research_areas: List[str]
  - contact_info: Dict
  - supporting_evidence: str
```

**Process**:
1. For each faculty match, generate explanation using LLM
2. Extract research areas and contact info
3. Create supporting evidence summary
4. Optionally generate email drafts

#### 3.4 Orchestrator

**Purpose**: Coordinate all agents in a unified workflow

**Capabilities**:
- Match proposal (PDF or text) → full recommendations
- Quick search (query string → recommendations)
- Generate email drafts for outreach
- Compare multiple proposals
- Manage conversation context

**Workflow**:
```
Input → Analyze → Retrieve → Recommend → Output
```

### 4. Model Layer

#### 4.1 LLM (Qwen2.5-Coder-32B-Instruct)

**Access**: HuggingFace Inference API
**Usage**:
- Proposal analysis and extraction
- Explanation generation
- Email draft creation

**Configuration**:
- Temperature: 0.2 (low for consistency)
- Max tokens: 512-1024
- Prompt engineering for structured output

#### 4.2 Embeddings (BAAI/bge-small-en-v1.5)

**Type**: Dense vector embeddings
**Dimension**: 384
**Usage**:
- Document embedding during ingestion
- Query embedding during retrieval

**Advantages**:
- Fast inference
- Good performance on academic text
- Small model size (~90MB)

#### 4.3 Vector Store (ChromaDB)

**Type**: Persistent vector database
**Features**:
- Efficient similarity search
- Metadata filtering
- Persistent storage

**Collections**:
- `faculty_all` (default): Combined CSV + PDF data

### 5. Configuration Management

**File**: `src/utils/config.py`

**Environment Variables**:
- `HF_TOKEN`: HuggingFace API token (required)
- `CHROMA_PATH`: Database location
- `EMBEDDING_MODEL`: Embedding model name
- `LLM_MODEL`: LLM model name
- `TOP_K_RESULTS`: Number of results to retrieve
- `CHUNK_SIZE`: Text chunk size
- `CHUNK_OVERLAP`: Overlap between chunks

## Data Flow

### Ingestion Phase

```
CSV File → load_faculty_csv() → Document objects
                                       ↓
PDF Files → load_pdfs_from_directory() →
                                       ↓
                            Combined Documents
                                       ↓
                            SentenceSplitter
                                       ↓
                            HuggingFaceEmbedding
                                       ↓
                            ChromaVectorStore
                                       ↓
                            Persisted in ChromaDB
```

### Query Phase

```
User Query/Proposal
        ↓
ProposalAnalyzer.analyze()
        ↓
ProposalAnalysis object
        ↓
FacultyRetriever.retrieve_from_analysis()
        ↓
Embedding → Similarity Search → Retrieve Nodes
        ↓
FacultyMatch objects (grouped by faculty)
        ↓
RecommendationAgent.generate_recommendations()
        ↓
For each match: LLM generates explanation
        ↓
FacultyRecommendation objects (ranked)
        ↓
Output: Recommendations + Report
```

## Design Patterns

### 1. Singleton Pattern

Models (LLM, embeddings) use singleton pattern to avoid redundant initialization:

```python
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = HuggingFaceInferenceAPI(...)
    return _llm
```

### 2. Builder Pattern

IndexManager provides fluent interface for complex index operations:

```python
manager = IndexManager()
manager.query("text")
manager.retrieve("text")
```

### 3. Factory Pattern

Multiple query engine creation modes:

```python
create_query_engine(mode="compact")
create_query_engine(mode="tree_summarize")
```

### 4. Facade Pattern

Orchestrator provides simplified interface to complex multi-agent workflow:

```python
orchestrator.match_proposal_pdf(path)  # Handles all agents internally
```

## Extension Points

### Adding New Data Sources

1. Create loader in `src/ingestion/`
2. Add to ingestion pipeline
3. Define metadata schema

### Adding New Agents

1. Create agent class in `src/agents/`
2. Define input/output data classes
3. Integrate with orchestrator

### Custom Retrieval Logic

1. Extend `FacultyRetriever`
2. Implement custom ranking/filtering
3. Use custom query engines

### UI Integration

1. Import `ResearchMatchOrchestrator`
2. Call `match_proposal_*()` methods
3. Display `MatchingResult` objects

## Performance Considerations

### Ingestion
- **Bottleneck**: Embedding generation
- **Optimization**: Batch processing, async operations
- **Cost**: ~1-2 seconds per document

### Retrieval
- **Bottleneck**: ChromaDB similarity search
- **Optimization**: Index optimization, proper chunk size
- **Cost**: ~100-500ms for top-k search

### LLM Calls
- **Bottleneck**: API latency
- **Optimization**: Caching, prompt optimization, lower temperature
- **Cost**: ~1-3 seconds per generation

## Security & Privacy

### Data Storage
- Local ChromaDB (no external storage)
- Configurable data retention

### API Access
- HuggingFace API requires token
- Token stored in `.env` (not committed)

### Sensitive Data
- Faculty information: Public data only
- Proposals: Not persisted by default
- User data: No tracking or collection

## Deployment Options

### Local Development
```bash
python scripts/ingest.py
python scripts/query_demo.py
```

### API Server
```python
# Future: FastAPI wrapper
app = FastAPI()
orchestrator = ResearchMatchOrchestrator()

@app.post("/match")
def match(proposal: str):
    return orchestrator.match_proposal_text(proposal)
```

### Web UI
```python
# Future: Streamlit/Gradio
import streamlit as st
result = orchestrator.match_proposal_text(st.text_area("Proposal"))
st.write(result.recommendations)
```

### GADK Deployment
```python
# Google Agent Development Kit integration
# For production multi-agent deployment
```

## Monitoring & Debugging

### Logs
- Console output with progress indicators
- Error traces with context

### Statistics
```python
from src.indexing.vector_store import get_collection_stats
stats = get_collection_stats()
```

### Testing
```bash
python scripts/test_system.py  # Component tests
python scripts/demo.py          # End-to-end demo
```

## Future Enhancements

1. **Caching**: Cache LLM responses for repeated queries
2. **Re-ranking**: Add learned re-ranking model
3. **Feedback Loop**: Incorporate user feedback to improve matching
4. **Multi-modal**: Support images in proposals
5. **Real-time**: WebSocket-based streaming responses
6. **Analytics**: Track query patterns and faculty engagement

