# Architecture Documentation

## 1. System Overview

The Faculty Research Matchmaker is an intelligent multi-agent recommendation system that matches PhD research proposals to relevant faculty members at the University of Oklahoma. The system uses semantic search, natural language processing, and multi-agent orchestration to provide personalized faculty recommendations with detailed explanations.

### Key Technologies

- **LlamaIndex**: Document processing, indexing, and retrieval framework
- **ChromaDB**: Vector database for storing and querying faculty embeddings
- **Qwen2.5-Coder-1.5B-Instruct**: Local LLM for reasoning and text generation
- **BGE Embeddings (BAAI/bge-small-en-v1.5)**: Semantic embeddings for vector search
- **Streamlit**: Web interface for user interaction
- **LangChain**: Multi-agent orchestration patterns

### Purpose

The system helps PhD candidates identify and connect with faculty members whose research interests align with their proposals. It analyzes proposals, searches through faculty data (both structured CSV profiles and unstructured PDF documents), and generates ranked recommendations with natural language explanations.

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                           │
│                         (Streamlit)                              │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ PDF Upload   │  │ Text Input   │  │ Quick Search │            │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘            │
│         │                 │                 │                    │
│         └─────────────────┴─────────────────┘                    │
│                            │                                     │
└────────────────────────────┼─────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Layer                           │
│              ResearchMatchOrchestrator                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Coordinates: Analysis → Retrieval → Recommendation      │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Proposal         │ │ Faculty          │ │ Recommendation   │
│ Analyzer Agent   │ │ Retriever Agent  │ │ Agent            │
│                  │ │                  │ │                  │
│ - Extract        │ │ - Semantic       │ │ - Rank matches   │
│   structured     │ │   search         │ │ - Generate       │
│   info           │ │ - Dual-store     │ │   explanations   │
│ - Generate       │ │   query          │ │ - Extract        │
│   search query   │ │ - Enrich with    │ │   metadata       │
│                  │ │   PDF evidence   │ │ - Email drafts   │
└──────────────────┘ └────────┬─────────┘ └──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Storage Layer                         │
│                         ChromaDB                                │
│                                                                 │
│  ┌────────────────────────┐  ┌────────────────────────┐         │
│  │ faculty_profiles       │  │ faculty_pdfs           │         │
│  │ (Structured CSV data)  │  │ (Unstructured PDFs)    │         │
│  │                        │  │                        │         │
│  │ - Name, email, website │  │ - CVs, papers          │         │
│  │ - Research areas       │  │ - Research summaries   │         │
│  │ - Department           │  │ - Metadata (LLM-extr.) │         │
│  └────────────────────────┘  └────────────────────────┘         │
│                                                                 │
│         Linked via faculty_id metadata                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Model Layer                              │
│                                                                 │
│  ┌───────────────────────┐  ┌───────────────────────┐           │
│  │ Qwen2.5-Coder-1.5B    │  │ BGE Embeddings        │           │
│  │ (LLM)                 │  │ (bge-small-en-v1.5)   │           │
│  │                       │  │                       │           │
│  │ - Analysis            │  │ - Vectorization       │           │
│  │ - Explanations        │  │ - Semantic search     │           │
│  │ - Email generation    │  │                       │           │
│  └───────────────────────┘  └───────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Core Components

### 3.1 Frontend Layer (Streamlit)

**File**: `app.py`

The Streamlit frontend provides a user-friendly web interface for interacting with the recommendation system.

#### Components

- **Session State Management**:
  - Caches the `ResearchMatchOrchestrator` instance to avoid re-initialization
  - Stores matching results for display
  - Manages user preferences (e.g., number of recommendations)

- **Three Input Modes**:
  1. **PDF Upload Tab**: Users can upload proposal PDFs via drag-and-drop
  2. **Text Input Tab**: Users can paste proposal text directly
  3. **Quick Search Tab**: Users can search by keywords or research areas

- **Results Display**:
  - Expandable faculty recommendation cards
  - Match scores and explanations
  - Research areas and contact information
  - Proposal analysis summary (when available)
  - Email draft generation UI

#### Integration Points

- **Initialization**: The orchestrator is initialized lazily on first use via `init_orchestrator()` function, which checks `st.session_state.orchestrator`
- **Method Calls**:
  - `orchestrator.match_proposal_pdf(pdf_path, generate_report=False)` for PDF uploads
  - `orchestrator.match_proposal_text(text, generate_report=False)` for text input
  - `orchestrator.quick_search(query, top_n)` for quick searches
  - `orchestrator.generate_email_for_recommendation(recommendation, analysis, sender_name)` for email generation
- **Data Display**: Results are displayed from `MatchingResult` objects stored in `st.session_state.results`

### 3.2 Orchestrator Layer

**File**: `src/agents/orchestrator.py`

**Class**: `ResearchMatchOrchestrator`

The orchestrator is the central coordinator that manages the entire matching workflow. It provides a unified API for the frontend and coordinates the three specialized agents.

#### Responsibilities

- Coordinates the three-stage workflow: analysis → retrieval → recommendation
- Manages three agent instances (ProposalAnalyzer, FacultyRetriever, RecommendationAgent)
- Provides a clean, unified API for the frontend
- Handles result aggregation and formatting

#### Key Methods

- **`match_proposal_pdf(pdf_path, generate_report=True)`**:
  - Complete workflow for PDF proposals
  - Returns `MatchingResult` with analysis, matches, and recommendations

- **`match_proposal_text(text, generate_report=True)`**:
  - Complete workflow for text proposals
  - Returns `MatchingResult` with analysis, matches, and recommendations

- **`quick_search(query, top_n=None)`**:
  - Simplified search without full proposal analysis
  - Returns `List[FacultyRecommendation]` directly

- **`generate_email_for_recommendation(recommendation, analysis, sender_name)`**:
  - Generates email drafts for contacting faculty
  - Returns formatted email text

- **`compare_proposals(proposals)`**:
  - Batch processing for multiple proposals
  - Returns dictionary of results keyed by proposal filename

#### Data Structures

- **`MatchingResult`**: Dataclass containing:
  - `proposal_analysis`: `ProposalAnalysis` object
  - `faculty_matches`: `List[FacultyMatch]` from retrieval
  - `recommendations`: `List[FacultyRecommendation]` from recommendation agent

### 3.3 Agent Layer

The system uses three specialized agents, each with a single responsibility.

#### 3.3.1 Proposal Analyzer Agent

**File**: `src/agents/proposal_analyzer.py`

**Class**: `ProposalAnalyzer`

The Proposal Analyzer extracts structured information from PhD proposals using LLM-based analysis.

##### Responsibilities

- Parse PDF or text input
- Extract structured information (topics, methods, domain, etc.)
- Generate search queries from analysis
- Handle text truncation for long documents

##### Output: `ProposalAnalysis` Dataclass

```python
@dataclass
class ProposalAnalysis:
    topics: List[str]              # Main research topics
    methods: List[str]             # Key methods/techniques
    domain: str                    # Primary research domain
    application_areas: List[str]   # Potential applications
    key_phrases: List[str]         # Distinctive phrases
    summary: str                   # 2-3 sentence summary
    full_text: str                 # Original text
```

##### LLM Usage

- Uses Qwen2.5-Coder to extract structured data via prompt engineering
- Prompt instructs LLM to format response with specific fields (TOPICS, METHODS, DOMAIN, etc.)
- Response is parsed line-by-line to extract structured information
- Handles text truncation (max 8000 chars) to fit model context

##### Key Methods

- `analyze_pdf(pdf_path)`: Loads PDF, extracts text, analyzes with LLM
- `analyze_text(text)`: Analyzes text directly
- `_analyze_text(text)`: Internal method that creates prompt and parses response
- `to_search_query()`: Converts analysis to formatted search query string

#### 3.3.2 Faculty Retriever Agent

**File**: `src/agents/faculty_retriever.py`

**Class**: `FacultyRetriever`

The Faculty Retriever performs semantic search over faculty data using a dual-store architecture.

##### Responsibilities

- Semantic search over faculty data
- Query both `faculty_profiles` and `faculty_pdfs` collections
- Enrich profile matches with PDF evidence
- Group and rank matches by faculty

##### Dual-Store Architecture

The system uses two separate ChromaDB collections:

1. **Primary: `faculty_profiles` Collection**
   - Contains structured CSV data
   - Fields: name, email, website, department, research areas, etc.
   - Used for initial ranking and fast retrieval
   - Each document has a `faculty_id` for linking

2. **Secondary: `faculty_pdfs` Collection**
   - Contains unstructured PDF content
   - Includes CVs, research papers, proposals
   - Enriched with LLM-extracted metadata (summary, research_interests, faculty_name)
   - Linked to profiles via `faculty_id` metadata

##### Retrieval Process

1. Query `faculty_profiles` collection with search query (from analysis or direct query)
2. Retrieve top-k profile matches
3. For each match, query `faculty_pdfs` collection filtered by `faculty_id`
4. Enrich each match with PDF evidence (summaries, research interests)
5. Return `FacultyMatch` objects with scores and supporting evidence

##### Output: `List[FacultyMatch]`

```python
@dataclass
class FacultyMatch:
    score: float                    # Similarity score
    faculty_name: Optional[str]     # Faculty name
    text: str                       # Matched text snippet
    metadata: Dict[str, Any]        # Full metadata
    source_type: str                # 'csv' or 'pdf'
    pdf_support: Optional[List[Dict]]  # Supporting PDF information
```

##### Key Methods

- `retrieve_from_analysis(analysis)`: Retrieves based on `ProposalAnalysis` object
- `retrieve_from_query(query, top_k)`: Direct query-based retrieval
- `_node_to_faculty_match(node)`: Converts retrieved nodes to `FacultyMatch` objects
- `_extract_pdf_support(pdf_node)`: Extracts summary and research interests from PDF nodes

#### 3.3.3 Recommendation Agent

**File**: `src/agents/recommender.py`

**Class**: `RecommendationAgent`

The Recommendation Agent ranks faculty matches and generates natural language explanations.

##### Responsibilities

- Rank and explain faculty matches
- Generate personalized natural language explanations
- Extract contact information and research areas
- Create email drafts for faculty contact
- Generate summary reports

##### Output: `List[FacultyRecommendation]`

```python
@dataclass
class FacultyRecommendation:
    rank: int                       # Ranking (1, 2, 3, ...)
    faculty_name: str               # Faculty name
    score: float                    # Match score
    explanation: str                # Natural language explanation
    research_areas: List[str]       # Extracted research areas
    contact_info: Dict[str, str]    # Email, website, department
    supporting_evidence: str       # Summary of evidence
```

##### LLM Usage

- Generates personalized explanations for each match
- Uses proposal analysis and faculty information to create context-aware explanations
- Incorporates PDF support information when available
- Generates professional email drafts

##### Key Methods

- `generate_recommendations(matches, analysis, top_n)`: Main method that ranks and explains matches
- `_generate_explanation(match, analysis)`: Creates LLM prompt and generates explanation
- `_extract_research_areas(match)`: Extracts research areas from metadata
- `_extract_contact_info(match)`: Extracts contact information
- `_create_supporting_evidence(match)`: Summarizes evidence from CSV and PDFs
- `generate_email_draft(recommendation, analysis, sender_name)`: Generates email text
- `generate_summary_report(recommendations, analysis)`: Creates formatted report

### 3.4 Data Storage Layer

#### 3.4.1 ChromaDB Vector Store

**File**: `src/indexing/vector_store.py`

ChromaDB provides persistent vector storage for faculty data using a dual-collection architecture.

##### Architecture: Dual-Collection System

- **`faculty_profiles`**: Structured CSV data
  - Faculty names, roles, departments
  - Research areas, interests
  - Contact information (email, website)
  - Each document tagged with `faculty_id` and `source='csv'`

- **`faculty_pdfs`**: Unstructured PDF content
  - Full text from CVs, papers, proposals
  - LLM-extracted metadata (summary, research_interests, faculty_name)
  - Each document tagged with `faculty_id` and `source='pdf'`
  - File names and document types stored in metadata

##### Integration

- Uses LlamaIndex `ChromaVectorStore` wrapper for seamless integration
- BGE embeddings generated during ingestion
- Persistent file-based storage at `Config.CHROMA_PATH` (default: `./faculty_chroma_db`)

##### Key Functions

- `get_chroma_client()`: Singleton ChromaDB client
- `get_or_create_collection(collection_name, reset)`: Collection management
- `get_vector_store(collection_name, reset)`: LlamaIndex-compatible vector store
- `get_collection_stats(collection_name)`: Statistics and metadata

#### 3.4.2 Index Management

**File**: `src/indexing/index_builder.py`

**Class**: `IndexManager`

The IndexManager handles the creation and management of LlamaIndex `VectorStoreIndex` instances.

##### Responsibilities

- Build `VectorStoreIndex` from ChromaDB collections
- Create retrievers for semantic search
- Manage query engines (optional, for LLM synthesis)
- Lazy initialization of indexes

##### Key Components

- **`VectorStoreIndex`**: LlamaIndex index built from ChromaDB vector store
- **`VectorIndexRetriever`**: Retriever for semantic search without LLM
- **`RetrieverQueryEngine`**: Query engine with LLM synthesis (optional)

##### Key Methods

- `build_index(collection_name)`: Builds index from collection
- `create_retriever(index, similarity_top_k)`: Creates retriever for search
- `create_query_engine(index, response_mode, similarity_top_k, llm)`: Creates query engine
- `IndexManager.index`: Property that lazily builds index
- `IndexManager.retriever`: Property that lazily creates retriever

### 3.5 Model Layer

#### 3.5.1 LLM (Qwen2.5-Coder)

**File**: `src/models/llm.py`

**Model**: Qwen/Qwen2.5-Coder-1.5B-Instruct (configurable via `Config.LLM_MODEL`)

The LLM is used for three main purposes:

1. **Proposal Analysis**: Structured extraction of topics, methods, domain, etc.
2. **Recommendation Explanations**: Natural language generation explaining why a faculty member is recommended
3. **Email Draft Generation**: Professional email composition

##### Configuration

- **Temperature**: Configurable via `Config.LLM_TEMPERATURE` (default: 0.2)
- **Max Tokens**: Configurable via `Config.LLM_MAX_TOKENS` (default: 512)
- **8-bit Quantization**: Optional via `Config.USE_8BIT_QUANTIZATION` (GPU only)
- **Device**: Auto-detect or force CPU/GPU via `Config.LLM_DEVICE`

##### Usage Patterns

- **Analysis**: Lower temperature (0.2) for structured extraction
- **Explanations**: Moderate temperature (0.3) for natural language
- **Email Generation**: Moderate temperature for professional tone

#### 3.5.2 Embeddings (BGE)

**File**: `src/models/embeddings.py`

**Model**: BAAI/bge-small-en-v1.5

BGE (BAAI General Embedding) provides semantic embeddings for vector search.

##### Usage

- Vectorization of faculty profiles (CSV data)
- Vectorization of PDF content
- Vectorization of proposal queries
- All text is embedded using the same model for consistent similarity search

##### Integration

- Used during ingestion to create embeddings for all documents
- Used during retrieval to embed queries
- ChromaDB stores embeddings and performs cosine similarity search

### 3.6 Ingestion Pipeline

**File**: `src/ingestion/pipeline.py`

The ingestion pipeline processes raw data (CSV and PDFs) and stores it in ChromaDB.

#### Process Flow

1. **Load CSV Data**:
   - Parse `DSAI-Faculties.csv`
   - Extract faculty profiles with metadata
   - Create documents with `source='csv'` and `faculty_id`

2. **Load PDF Files**:
   - Extract text from PDFs in `data/pdfs/`
   - Optionally use LLM to extract metadata (summary, research_interests, faculty_name)
   - Create documents with `source='pdf'` and `faculty_id`

3. **Chunk Documents**:
   - Split long documents using `SentenceSplitter`
   - Chunk size: `Config.CHUNK_SIZE` (default: 1024)
   - Overlap: `Config.CHUNK_OVERLAP` (default: 128)

4. **Generate Embeddings**:
   - Use BGE embedding model
   - Create embeddings for all chunks

5. **Store in ChromaDB**:
   - Store in appropriate collection (`faculty_profiles` or `faculty_pdfs`)
   - Batch processing with `Config.INGESTION_BATCH_SIZE` (default: 5000)

#### Key Components

- **`run_ingestion_pipeline()`**: Main async function
- **`_ingest_documents_to_collection()`**: Ingests documents to a specific collection
- PDF metadata extraction can be disabled via `Config.EXTRACT_PDF_METADATA_WITH_LLM`

## 4. Data Flow

### 4.1 Full Proposal Matching Flow

This flow is triggered when a user uploads a PDF or enters proposal text.

```
1. User Action
   └─> Streamlit frontend receives PDF upload or text input

2. Frontend Processing
   └─> Saves PDF to temporary file (if PDF)
   └─> Calls orchestrator.match_proposal_pdf() or match_proposal_text()

3. Orchestrator → ProposalAnalyzer
   ├─> Analyzes proposal with LLM
   ├─> Extracts: topics, methods, domain, application_areas, key_phrases, summary
   └─> Returns ProposalAnalysis object

4. Orchestrator → FacultyRetriever
   ├─> Converts ProposalAnalysis to search query
   ├─> Queries faculty_profiles collection (primary ranking)
   │   └─> Retrieves top-k profile matches with scores
   ├─> For each match, queries faculty_pdfs collection
   │   ├─> Filters by faculty_id
   │   └─> Retrieves supporting PDF evidence
   └─> Returns List[FacultyMatch] with PDF support

5. Orchestrator → RecommendationAgent
   ├─> Takes matches and analysis
   ├─> For each match:
   │   ├─> Generates explanation with LLM
   │   ├─> Extracts research areas and contact info
   │   └─> Creates supporting evidence summary
   └─> Returns List[FacultyRecommendation]

6. Results Aggregation
   └─> Orchestrator creates MatchingResult object
       ├─> proposal_analysis: ProposalAnalysis
       ├─> faculty_matches: List[FacultyMatch]
       └─> recommendations: List[FacultyRecommendation]

7. Frontend Display
   └─> Displays recommendations with expandable cards
       ├─> Shows proposal analysis (if available)
       ├─> Shows ranked faculty with explanations
       ├─> Displays contact information
       └─> Provides email draft generation
```

### 4.2 Quick Search Flow

This simplified flow is used for keyword-based searches without full proposal analysis.

```
1. User Action
   └─> User enters search query in Quick Search tab

2. Frontend Processing
   └─> Calls orchestrator.quick_search(query, top_n)

3. Orchestrator → FacultyRetriever
   ├─> Direct query to faculty_profiles collection
   ├─> Retrieves top-k matches
   └─> Enriches with PDF evidence (same as full flow)

4. Orchestrator → RecommendationAgent
   ├─> Creates minimal ProposalAnalysis from query
   ├─> Generates recommendations with explanations
   └─> Returns List[FacultyRecommendation]

5. Frontend Display
   └─> Displays recommendations (no proposal analysis shown)
```

## 5. Key Design Patterns

### 5.1 Dual-Store Architecture

The system separates structured and unstructured data into two collections:

- **Separation of Concerns**: CSV profiles for fast ranking, PDFs for detailed evidence
- **Performance**: Profile collection is smaller and faster to query
- **Richness**: PDF collection provides deep context when needed
- **Linking**: `faculty_id` metadata connects profiles to PDFs

This architecture allows the system to:
- Quickly rank faculty based on structured data
- Provide detailed evidence from PDFs when available
- Handle cases where PDFs are missing gracefully

### 5.2 Agent Orchestration

The orchestrator pattern provides:

- **Single Responsibility**: Each agent has one clear purpose
- **Coordination**: Orchestrator manages workflow and data flow
- **Unified API**: Frontend interacts with one interface
- **Flexibility**: Agents can be swapped or modified independently

### 5.3 Lazy Initialization

Components are initialized on-demand:

- **Models**: LLM and embeddings loaded on first use
- **Indexes**: Built when first accessed
- **Orchestrator**: Cached in Streamlit session state
- **Benefits**: Faster startup, lower memory usage when not in use

## 6. Configuration

**File**: `src/utils/config.py`

**Class**: `Config`

All configuration is centralized and environment-based via `.env` file.

### Key Settings

#### Model Configuration
- `LLM_MODEL`: LLM model path (default: "Qwen/Qwen2.5-Coder-1.5B-Instruct")
- `EMBEDDING_MODEL`: Embedding model (default: "BAAI/bge-small-en-v1.5")
- `LLM_TEMPERATURE`: LLM temperature (default: 0.2)
- `LLM_MAX_TOKENS`: Max tokens per generation (default: 512)
- `USE_8BIT_QUANTIZATION`: Enable 8-bit quantization for GPU (default: false)
- `LLM_DEVICE`: Device selection - "auto", "cuda", or "cpu" (default: "auto")

#### ChromaDB Configuration
- `CHROMA_PATH`: Path to ChromaDB storage (default: "./faculty_chroma_db")
- `COLLECTION_NAME`: Default collection name (default: "faculty_all")
- `FACULTY_PROFILES_COLLECTION`: Profiles collection (default: "faculty_profiles")
- `FACULTY_PDFS_COLLECTION`: PDFs collection (default: "faculty_pdfs")

#### Retrieval Configuration
- `TOP_K_RESULTS`: Default number of results (default: 10)
- `CHUNK_SIZE`: Text chunk size (default: 1024)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 128)

#### Ingestion Configuration
- `INGESTION_BATCH_SIZE`: Batch size for ingestion (default: 5000)
- `EXTRACT_PDF_METADATA_WITH_LLM`: Enable LLM-based PDF metadata extraction (default: true)

#### Data Paths
- `DATA_DIR`: Data directory (default: "data")
- `CSV_PATH`: Path to CSV file (default: "data/DSAI-Faculties.csv")
- `PDF_DIR`: PDF directory (default: "data/pdfs")

#### Required
- `HF_TOKEN`: HuggingFace token (required for model downloads)

## 7. File Structure Reference

```
dsai-recommender/
├── app.py                          # Streamlit frontend
├── Architecture.md                 # This file
├── README.md                       # User documentation
├── requirements.txt                # Python dependencies
├── .env.template                   # Configuration template
│
├── src/
│   ├── agents/                     # Multi-agent system
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # Main coordinator
│   │   ├── proposal_analyzer.py    # Analysis agent
│   │   ├── faculty_retriever.py    # Retrieval agent
│   │   └── recommender.py           # Recommendation agent
│   │
│   ├── indexing/                   # Vector store management
│   │   ├── __init__.py
│   │   ├── vector_store.py         # ChromaDB integration
│   │   └── index_builder.py         # Index creation
│   │
│   ├── ingestion/                  # Data loading
│   │   ├── __init__.py
│   │   ├── pipeline.py             # Main ingestion flow
│   │   ├── csv_loader.py            # CSV processing
│   │   ├── pdf_loader.py            # PDF processing
│   │   └── enrichment.py            # Metadata enrichment
│   │
│   ├── models/                     # AI models
│   │   ├── __init__.py
│   │   ├── llm.py                  # LLM setup
│   │   └── embeddings.py           # Embedding model
│   │
│   ├── utils/                      # Configuration and utilities
│   │   ├── __init__.py
│   │   ├── config.py               # Centralized config
│   │   ├── faculty_id.py            # Faculty ID generation
│   │   └── name_matcher.py         # Name matching utilities
│   │
│   └── web/                        # Web UI components (optional)
│       ├── __init__.py
│       └── ui_components.py
│
├── scripts/
│   ├── ingest.py                   # Data ingestion script
│   ├── match.py                    # CLI matching interface
│   ├── run_app.py                  # Streamlit launcher
│   └── ...
│
├── data/
│   ├── DSAI-Faculties.csv          # Faculty metadata
│   └── pdfs/                       # Faculty PDFs
│
└── faculty_chroma_db/               # ChromaDB storage (generated)
    ├── faculty_profiles/           # Profiles collection
    └── faculty_pdfs/                # PDFs collection
```

## 8. Integration Points

### 8.1 Frontend → Orchestrator

**Entry Point**: `ResearchMatchOrchestrator` initialization in `app.py`

**Initialization**:
```python
orchestrator = ResearchMatchOrchestrator(
    top_n_recommendations=st.session_state.top_n
)
```

**Methods Called**:
- `match_proposal_pdf(pdf_path, generate_report=False)` → Returns `MatchingResult`
- `match_proposal_text(text, generate_report=False)` → Returns `MatchingResult`
- `quick_search(query, top_n)` → Returns `List[FacultyRecommendation]`
- `generate_email_for_recommendation(recommendation, analysis, sender_name)` → Returns `str`

**Data Flow**:
- Frontend passes user input (PDF path or text)
- Orchestrator returns structured results
- Frontend displays results using Streamlit components

### 8.2 Orchestrator → Agents

**ProposalAnalyzer**:
- `analyze_pdf(pdf_path)` → Returns `ProposalAnalysis`
- `analyze_text(text)` → Returns `ProposalAnalysis`

**FacultyRetriever**:
- `retrieve_from_analysis(analysis)` → Returns `List[FacultyMatch]`
- `retrieve_from_query(query, top_k)` → Returns `List[FacultyMatch]`

**RecommendationAgent**:
- `generate_recommendations(matches, analysis, top_n)` → Returns `List[FacultyRecommendation]`
- `generate_email_draft(recommendation, analysis, sender_name)` → Returns `str`

### 8.3 Agents → Storage

**IndexManager**:
- `IndexManager(collection_name)` → Creates manager for collection
- `manager.retriever.retrieve(query)` → Performs semantic search
- `manager.index` → Accesses underlying VectorStoreIndex

**ChromaDB**:
- Accessed via LlamaIndex `ChromaVectorStore` wrapper
- Collections: `faculty_profiles` and `faculty_pdfs`
- Metadata filtering via `MetadataFilters` and `ExactMatchFilter`

## 9. Error Handling

The system includes several error handling mechanisms:

### Frontend Layer
- **Session State Validation**: Checks for orchestrator existence before use
- **Collection Status**: Validates ChromaDB collections on startup
- **File Handling**: Temporary file cleanup for PDF uploads
- **User Feedback**: Error messages displayed via Streamlit components

### Agent Layer
- **Model Initialization**: Try-catch blocks around LLM/embedding loading
- **PDF Processing**: Handles cases where PDFs cannot be loaded
- **Metadata Extraction**: Graceful degradation when PDF metadata unavailable
- **Empty Results**: Handles cases where no matches are found

### Storage Layer
- **Collection Existence**: Checks for collection existence before querying
- **Connection Errors**: Handles ChromaDB connection issues
- **Missing Metadata**: Defaults for missing faculty_id or other metadata

## 10. Performance Considerations

### Optimization Strategies

1. **Model Caching**:
   - LLM and embeddings loaded once and cached
   - Orchestrator cached in Streamlit session state
   - Reduces initialization overhead

2. **Lazy Loading**:
   - Models loaded on first use, not at import
   - Indexes built on demand
   - Reduces startup time significantly

3. **Batch Processing**:
   - Ingestion uses configurable batch sizes
   - Prevents memory issues with large datasets

4. **Configurable Retrieval**:
   - `top_k` parameter controls result count
   - Users can adjust number of recommendations
   - Reduces processing time for smaller result sets

5. **Optional PDF Metadata**:
   - LLM-based PDF metadata extraction can be disabled
   - Reduces processing time during ingestion
   - Controlled via `Config.EXTRACT_PDF_METADATA_WITH_LLM`

6. **Dual-Store Efficiency**:
   - Profile collection is smaller and faster
   - PDF evidence only retrieved for top matches
   - Reduces query time while maintaining richness

### Performance Characteristics

- **Startup Time**: ~2-5 seconds (model loading)
- **PDF Analysis**: ~5-15 seconds (depends on document length and LLM speed)
- **Retrieval**: ~1-3 seconds (semantic search)
- **Recommendation Generation**: ~3-10 seconds per recommendation (LLM generation)
- **Total Workflow**: ~15-45 seconds for full proposal matching

*Note: Times are approximate and depend on hardware, model size, and document complexity.*

## 11. Future Enhancements

Potential areas for improvement:

1. **Caching**: Cache proposal analyses to avoid re-processing
2. **Async Processing**: Make more operations async for better concurrency
3. **Batch Recommendations**: Generate multiple recommendations in parallel
4. **Advanced Filtering**: Add filters by department, research area, etc.
5. **Feedback Loop**: Learn from user interactions to improve recommendations
6. **Multi-modal**: Support for images, tables, and other document types
7. **API Endpoints**: REST API for programmatic access
8. **Real-time Updates**: WebSocket support for live updates

---

*This architecture documentation is maintained alongside the codebase. For implementation details, refer to the source files mentioned in each section.*

