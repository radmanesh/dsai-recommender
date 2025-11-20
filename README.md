# Agentic Research Matchmaker

An intelligent recommendation system that matches Ph.D. proposals to OU faculty using semantic search and multi-agent orchestration.

## Overview

This system uses:
- **LlamaIndex** for document processing and indexing
- **ChromaDB** for vector storage
- **Qwen2.5-1.5B-Instruct** for reasoning (runs locally)
- **HuggingFace BGE embeddings** (BAAI/bge-small-en-v1.5) for semantic search
- **LangChain** for multi-agent orchestration
- **requests & BeautifulSoup4** for website crawling

## Features

- 📄 Ingest faculty data from CSV and PDFs (CVs, papers, proposals)
- 🌐 Automatic website crawling for faculty and lab websites
- 🔍 Semantic search over structured and unstructured faculty information
- 🤖 Multi-agent system for proposal analysis and faculty matching
- 💡 Natural language explanations for recommendations
- 📊 Ranked faculty recommendations based on research alignment

## Architecture

### System Overview

The Faculty Research Matchmaker uses a multi-agent orchestration system built on semantic search and local LLM reasoning. The architecture consists of:

#### Core Components

1. **Orchestrator** (`ResearchMatchOrchestrator`): Coordinates the entire matching workflow
2. **Proposal Analyzer** (`ProposalAnalyzer`): Extracts key information from PhD proposals using LLM
3. **Faculty Retriever** (`FacultyRetriever`): Performs semantic search across multiple ChromaDB collections
4. **Recommendation Agent** (`RecommendationAgent`): Ranks faculty and generates explanations

#### Data Layers

- **Vector Store (ChromaDB)**: Three separate collections for different data types
  - `faculty_profiles`: Faculty metadata from CSV
  - `faculty_pdfs`: PDF documents (CVs, papers, proposals)
  - `faculty_websites`: Crawled website content

- **Embeddings**: BGE (BAAI/bge-small-en-v1.5) for semantic search
- **LLM**: Qwen2.5-1.5B-Instruct for analysis, explanations, and email generation

#### Workflow

```
User Input (PDF/Text/Query)
    ↓
Orchestrator
    ↓
[Step 1] Proposal Analyzer → Extract topics, methods, domain
    ↓
[Step 2] Faculty Retriever → Semantic search across collections
    ↓
[Step 3] Recommendation Agent → Rank and explain matches
    ↓
Results (Ranked Faculty Recommendations)
```

### Framework Structure

```
src/
├── agents/              # Multi-agent system
│   ├── orchestrator.py      # Main workflow coordinator
│   ├── proposal_analyzer.py # Proposal analysis using LLM
│   ├── faculty_retriever.py # Semantic search agent
│   └── recommender.py       # Ranking and explanation agent
├── ingestion/           # Data loading and processing
│   ├── csv_loader.py        # CSV faculty data loading
│   ├── pdf_loader.py        # PDF document processing
│   ├── website_crawler.py   # Website crawling (requests + BeautifulSoup4)
│   ├── enrichment.py        # PDF-to-CSV enrichment
│   └── pipeline.py          # Main ingestion pipeline
├── indexing/            # Vector store management
│   ├── vector_store.py      # ChromaDB operations
│   └── index_builder.py     # Index construction
├── models/              # ML models
│   ├── llm.py              # LLM initialization and management
│   └── embeddings.py       # Embedding model setup
├── utils/               # Utilities
│   ├── config.py           # Configuration management
│   ├── logger.py           # Centralized logging
│   ├── faculty_id.py       # Faculty ID mapping
│   └── name_matcher.py     # Name matching utilities
└── web/                 # Web interface
    └── ui_components.py    # Streamlit UI components
```

## Project Structure

```
dsai-recommender/
├── app.py                           # Streamlit web interface
├── .env.template                    # Environment configuration template
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── QUICKSTART.md                    # Quick start guide
├── data/
│   ├── DSAI-Faculties.csv          # Faculty metadata
│   └── pdfs/                        # Faculty CVs, papers, proposals
├── src/
│   ├── ingestion/                   # Data loading and processing
│   ├── indexing/                    # Vector store and index management
│   ├── agents/                      # Multi-agent implementation
│   ├── models/                      # LLM and embedding setup
│   ├── web/                         # Web UI components
│   └── utils/                       # Configuration and utilities
├── scripts/
│   ├── ingest.py                    # Data ingestion script
│   ├── match.py                     # CLI matching interface
│   ├── query_demo.py                # Interactive query demo
│   ├── select_model.py              # Hardware checker & model selector
│   ├── test_local_model.py          # Model verification
│   └── run_app.py                   # Streamlit app launcher
└── notebooks/
    └── recommender.ipynb            # Development notebook
```

## Hardware Requirements

**Minimum (CPU Mode):**
- RAM: 8GB+ available
- Disk: 5GB free space
- CPU: Multi-core processor

**Recommended (GPU Mode):**
- GPU: 4GB+ VRAM (NVIDIA with CUDA)
- RAM: 16GB+
- Disk: 10GB free space

## Setup

### 1. Clone and Install

```bash
git clone <repository-url>
cd dsai-recommender
```

#### Option A: Using Mamba (Recommended)

Mamba is a faster drop-in replacement for conda:

```bash
# Create mamba environment
mamba create -n recommender python=3.10
mamba activate recommender

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using Conda

If you prefer conda:

```bash
# Create conda environment
conda create -n recommender python=3.10
conda activate recommender

# Install dependencies
pip install -r requirements.txt
```

#### Option C: Using Virtualenv

Alternative using Python's built-in venv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** After creating the environment, always activate it before running any scripts:
- Mamba: `mamba activate recommender`
- Conda: `conda activate recommender`
- Virtualenv: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)

### 2. Check Your Hardware

```bash
python scripts/select_model.py
```

This will show your system resources and recommend the best model size.

### 3. Configure Environment

Copy `.env.template` to `.env` and configure:

```bash
cp .env.template .env
```

Edit `.env` and add your HuggingFace token (needed for initial model download):
```
HF_TOKEN=your_actual_token_here
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct  # Default, runs on CPU
```

Get your token from: https://huggingface.co/settings/tokens

**Available Models:**
- `Qwen/Qwen2.5-1.5B-Instruct` - Best for CPU (~3GB)
- `Qwen/Qwen2.5-3B-Instruct` - Medium, needs GPU/good CPU (~6GB)
- `Qwen/Qwen2.5-7B-Instruct` - Large, needs GPU 8GB+ VRAM (~14GB)

### 4. Test Model Loading

```bash
python scripts/test_local_model.py
```

First run will download the model (~3GB). This is a one-time operation.

### 5. Prepare Data

Place your data files:
- `data/DSAI-Faculties.csv` - Faculty metadata (name, role, department, areas, research interests, website URLs, etc.)
- `data/pdfs/` - Faculty CVs, publications, etc. (PDFs should be organized here)

**Note:** The system will automatically crawl faculty and lab websites if URLs are provided in the CSV `Website` and `Lab Website` columns.

### 6. Ingest Data

**Important:** Make sure your conda/mamba environment is activated:
```bash
# For mamba
mamba activate recommender

# For conda
conda activate recommender
```

Run the ingestion pipeline to index all data:

```bash
python scripts/ingest.py
```

**What the ingestion pipeline does:**

1. **CSV Validation** - Validates required and recommended columns
2. **CSV Loading** - Loads faculty profiles with metadata
3. **PDF Loading** - Loads all PDFs from `data/pdfs/` directory
   - Extracts text from PDFs
   - Optionally uses LLM to extract metadata (summary, research interests, faculty name)
   - Maps faculty names to faculty IDs
4. **PDF Enrichment** (Optional) - Enriches CSV profiles with information from PDFs
5. **Website Crawling** (Optional) - Crawls faculty and lab websites
   - Only crawls pages within the same directory path as the starting URL
   - Stores crawled content with faculty ID metadata
6. **Vectorization** - Creates embeddings using BGE model
7. **Storage** - Stores in ChromaDB collections:
   - `faculty_profiles` - Faculty metadata from CSV (enriched with PDF data)
   - `faculty_pdfs` - PDF documents (CVs, papers, etc.)
   - `faculty_websites` - Crawled website pages

**Ingestion Options:**
- `--reset`: Reset collections before ingestion (via interactive prompt)
- Website crawling can be disabled by setting `enable_website_crawling=False` in the pipeline

## Usage

### Scripts Overview

The framework provides several scripts for different use cases:

| Script | Purpose | Usage |
|--------|---------|-------|
| `ingest.py` | Data ingestion and indexing | `python scripts/ingest.py` |
| `match.py` | CLI matching interface | `python scripts/match.py [options]` |
| `query_demo.py` | Interactive query testing | `python scripts/query_demo.py [--interactive]` |
| `inspect_nodes.py` | Inspect vector store contents | `python scripts/inspect_nodes.py [options]` |
| `select_model.py` | Hardware check & model selection | `python scripts/select_model.py` |
| `test_local_model.py` | Test LLM model loading | `python scripts/test_local_model.py` |
| `run_app.py` | Launch Streamlit web interface | `python scripts/run_app.py` |

### 1. Data Ingestion (`ingest.py`)

**Purpose:** Load and index faculty data from CSV, PDFs, and websites into ChromaDB.

**Usage:**
```bash
# Make sure environment is activated
mamba activate recommender  # or: conda activate recommender

# Run ingestion
python scripts/ingest.py
```

**What it does:**
1. Validates CSV format
2. Loads CSV faculty profiles
3. Loads PDF documents from `data/pdfs/`
4. Crawls faculty and lab websites (if URLs provided)
5. Creates embeddings and stores in ChromaDB collections:
   - `faculty_profiles`: CSV metadata
   - `faculty_pdfs`: PDF content
   - `faculty_websites`: Crawled website pages
6. Optionally enriches CSV profiles with PDF information

**Features:**
- Interactive confirmation before ingestion
- Preview of documents to be loaded
- Option to reset existing collections
- Comprehensive logging based on `DEBUG_LEVEL`

### 2. Command-Line Matching (`match.py`)

**Purpose:** Match proposals to faculty via CLI.

**Usage:**
```bash
# Match text proposal
python scripts/match.py "Research on multi-agent AI systems and LLM optimization..."

# Match PDF proposal
python scripts/match.py --file proposal.pdf

# Quick keyword search
python scripts/match.py --query "machine learning and NLP"

# Specify number of results
python scripts/match.py --query "reinforcement learning" --top 10

# Generate email draft for top match
python scripts/match.py --file proposal.pdf --email "John Doe"
```

**Options:**
- `text`: Proposal text as argument
- `--file, -f`: Path to proposal PDF
- `--query, -q`: Quick keyword search
- `--top, -n`: Number of recommendations (default: 5)
- `--email, -e`: Generate email draft (provide sender name)
- `--no-report`: Skip detailed report output

### 3. Interactive Query Demo (`query_demo.py`)

**Purpose:** Test queries interactively or run example queries.

**Usage:**
```bash
# Run example queries
python scripts/query_demo.py

# Interactive mode
python scripts/query_demo.py --interactive

# Single query
python scripts/query_demo.py --query "multi-agent systems"
```

**Features:**
- Pre-built example queries
- Interactive query mode
- Toggle between query (with LLM) and retrieval-only modes
- Display source nodes and scores

### 4. Inspect Vector Store (`inspect_nodes.py`)

**Purpose:** Inspect and debug contents of ChromaDB collections.

**Usage:**
```bash
# Inspect faculty_profiles collection (default)
python scripts/inspect_nodes.py

# Inspect specific collection
python scripts/inspect_nodes.py --collection faculty_pdfs

# Inspect with custom query
python scripts/inspect_nodes.py --query "machine learning" --top 5

# Show full text
python scripts/inspect_nodes.py --full-text

# List all nodes in collection
python scripts/inspect_nodes.py --list

# Check metadata schema
python scripts/inspect_nodes.py --schema
```

**Options:**
- `--collection`: Collection to inspect (`faculty_profiles`, `faculty_pdfs`, `faculty_websites`, or `both`)
- `--query, -q`: Query string
- `--top, -k`: Number of nodes to retrieve
- `--full-text`: Show full text instead of preview
- `--list`: List all node summaries
- `--schema`: Show metadata schema

### 5. Model Selection (`select_model.py`)

**Purpose:** Check system resources and get model recommendations.

**Usage:**
```bash
python scripts/select_model.py
```

**Output:**
- GPU/CPU detection
- RAM and VRAM information
- Model recommendations based on hardware
- Instructions for changing model

### 6. Test Model Loading (`test_local_model.py`)

**Purpose:** Verify LLM model can be loaded successfully.

**Usage:**
```bash
python scripts/test_local_model.py
```

**What it does:**
- Loads the configured LLM model
- Tests inference
- Reports download status
- Useful for troubleshooting model issues

### 7. Web Interface (`run_app.py` or `app.py`)

**Purpose:** Launch Streamlit web application for interactive use.

**Usage:**
```bash
# Method 1: Via launcher script
python scripts/run_app.py

# Method 2: Direct Streamlit
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8502
```

**Access at:** http://localhost:8501

**Features:**
- 📄 **Drag-and-drop PDF upload** - Upload proposal PDFs
- ✍️ **Text input** - Paste proposal text directly
- 🔍 **Quick search** - Search by keywords or research areas
- 📊 **Interactive results** - Expandable faculty cards with details
- ✉️ **Email drafts** - Auto-generate contact emails
- ⚙️ **Settings** - Adjust number of recommendations

### Programmatic Usage

Use the framework in your own Python code:

```python
from src.agents.orchestrator import ResearchMatchOrchestrator
from pathlib import Path

# Initialize orchestrator
orchestrator = ResearchMatchOrchestrator(top_n_recommendations=5)

# Method 1: Match PDF proposal
result = orchestrator.match_proposal_pdf("path/to/proposal.pdf")
for rec in result.recommendations:
    print(f"{rec.rank}. {rec.faculty_name}")
    print(f"   Score: {rec.score:.3f}")
    print(f"   Explanation: {rec.explanation}")
    print(f"   Contact: {rec.contact_info.get('email', 'N/A')}")

# Method 2: Match text proposal
result = orchestrator.match_proposal_text("Research on reinforcement learning...")

# Method 3: Quick search
recommendations = orchestrator.quick_search("machine learning", top_n=3)

# Method 4: Generate email
if result.recommendations:
    email = orchestrator.generate_email_for_recommendation(
        result.recommendations[0],
        result.proposal_analysis,
        sender_name="Your Name"
    )
    print(email)
```

## Data Format

### CSV Format (DSAI-Faculties.csv)

Required columns:
- `name`: Faculty name (required)
- `faculty_id`: Unique faculty identifier (required)

Recommended columns:
- `role`: Position (Professor, Associate Professor, etc.)
- `department`: Department name
- `areas`: Research areas (comma-separated)
- `research_interests`: Detailed research interests
- `website`: Faculty website URL (for automatic crawling)
- `lab_website` or `Lab Website`: Lab website URL (for automatic crawling)
- `email`: Contact email

**Website Crawling:**
- URLs from `website` and `lab_website` columns are automatically crawled
- Crawler only fetches pages within the same directory path as the starting URL
- Example: If `website` is `https://ou.edu/cashbarker`, only pages starting with `https://ou.edu/cashbarker` are fetched
- If URL is `https://ou.edu/cashbarker/index.html`, then `https://ou.edu/cashbarker/publication.html` is accepted
- All crawled content is stored in `faculty_websites` collection with `faculty_id` metadata

### PDF Files

PDFs should be placed in `data/pdfs/` directory with clear naming conventions:
- `{faculty_name}_CV.pdf` - Example: `John_Doe_CV.pdf`
- `{faculty_name}_paper_{title}.pdf` - Example: `Jane_Smith_paper_reinforcement_learning.pdf`
- `{faculty_name}_proposal.pdf` - Example: `Bob_Johnson_proposal.pdf`

**PDF Processing:**
- The system automatically extracts text from PDFs
- Faculty names are extracted from filenames or PDF content
- Faculty names are matched to CSV records via `faculty_id` mapping
- Optionally uses LLM to extract summary and research interests from PDFs

**Note:** Sample/test PDFs placed in the root `data/` directory (like `sample_proposal.pdf`) are not automatically ingested, as the PDF loader reads from `data/pdfs/` only.

## Configuration

Key configuration options in `.env`:

- `HF_TOKEN`: HuggingFace token (required for initial download)
- `LLM_MODEL`: Local LLM model to use (default: Qwen2.5-1.5B-Instruct)
- `USE_8BIT_QUANTIZATION`: Reduce memory usage by 50% (GPU only, default: false)
- `LLM_DEVICE`: Device to use (auto/cuda/cpu, default: auto)
- `CHROMA_PATH`: ChromaDB storage path
- `COLLECTION_NAME`: ChromaDB collection name
- `EMBEDDING_MODEL`: Embedding model name
- `TOP_K_RESULTS`: Number of results to retrieve
- `CHUNK_SIZE`: Text chunk size for splitting
- `CHUNK_OVERLAP`: Overlap between chunks
- `DEBUG_LEVEL`: Debug logging level (ERROR=0, WARNING=1, INFO=2, DEBUG=3, VERBOSE=4, default: INFO)

### Debug Levels

The system supports configurable debug logging levels set via `DEBUG_LEVEL` environment variable:

- **ERROR (0)**: Only error messages
- **WARNING (1)**: Warnings and errors
- **INFO (2)**: Informational messages, warnings, and errors (default)
- **DEBUG (3)**: Detailed debug information including processing steps, metadata, and intermediate results
- **VERBOSE (4)**: Very detailed output including timestamps, file paths, full metadata, and trace information

**Usage:**
```bash
# Set debug level in .env
DEBUG_LEVEL=DEBUG

# Or as integer
DEBUG_LEVEL=3

# Or via environment variable
export DEBUG_LEVEL=VERBOSE
python scripts/ingest.py
```

**What gets logged:**

- **INFO level**: Progress messages, completion status, summary statistics
- **DEBUG level**: Processing details, collection names, node counts, faculty IDs, query strings
- **VERBOSE level**: Full metadata, sample data, timestamps, file paths, exception details

## Deployment

### Local Development

```bash
streamlit run app.py
```

### Production (Wukong Server)

```bash
# Run in background with nohup
nohup streamlit run app.py --server.port 8501 --server.headless true &

# Or use screen/tmux
screen -S matchmaker
streamlit run app.py
# Ctrl+A, D to detach
```

### Local Network Access

```bash
streamlit run app.py --server.address 0.0.0.0
```

## Development

### Framework Components

#### Agents (`src/agents/`)

1. **Orchestrator** (`orchestrator.py`): Main coordinator
   - `match_proposal_pdf()`: Full workflow for PDF proposals
   - `match_proposal_text()`: Full workflow for text proposals
   - `quick_search()`: Fast keyword-based search
   - `generate_email_for_recommendation()`: Email generation

2. **Proposal Analyzer** (`proposal_analyzer.py`): Extracts proposal information
   - `analyze_pdf()`: Analyze PDF proposal
   - `analyze_text()`: Analyze text proposal
   - Returns: `ProposalAnalysis` with topics, methods, domain, etc.

3. **Faculty Retriever** (`faculty_retriever.py`): Semantic search agent
   - `retrieve_from_analysis()`: Search based on proposal analysis
   - `retrieve_from_query()`: Search based on query string
   - Searches across `faculty_profiles`, `faculty_pdfs`, and `faculty_websites` collections
   - Returns: `List[FacultyMatch]` with scores and metadata

4. **Recommendation Agent** (`recommender.py`): Ranking and explanation
   - `generate_recommendations()`: Rank and explain matches
   - `generate_summary_report()`: Create formatted report
   - `generate_email_draft()`: Generate contact email
   - Returns: `List[FacultyRecommendation]` with explanations

#### Ingestion (`src/ingestion/`)

1. **CSV Loader** (`csv_loader.py`): Loads faculty metadata
   - `load_faculty_csv()`: Load and parse CSV
   - `validate_csv_format()`: Validate CSV structure

2. **PDF Loader** (`pdf_loader.py`): Processes PDF documents
   - `load_pdfs_from_directory()`: Load all PDFs from directory
   - `load_single_pdf()`: Load a single PDF
   - `_extract_metadata_with_llm()`: Extract metadata using LLM

3. **Website Crawler** (`website_crawler.py`): Crawls faculty websites
   - `extract_urls_from_csv_docs()`: Extract URLs from CSV
   - `crawl_website()`: Crawl a single website (same directory level only)
   - `crawl_faculty_websites()`: Crawl all faculty websites

4. **Enrichment** (`enrichment.py`): Enriches CSV with PDF data
   - `enrich_csv_documents()`: Add PDF-derived information to CSV docs
   - `get_pdf_nodes_by_faculty()`: Query PDF collection by faculty ID

5. **Pipeline** (`pipeline.py`): Main ingestion orchestrator
   - `run_ingestion_pipeline()`: Complete ingestion workflow
   - `_ingest_documents_to_collection()`: Ingest to specific collection

#### Indexing (`src/indexing/`)

1. **Vector Store** (`vector_store.py`): ChromaDB operations
   - `get_vector_store()`: Get or create vector store
   - `get_or_create_collection()`: Get or create collection
   - `get_collection_stats()`: Get collection statistics

2. **Index Builder** (`index_builder.py`): Index management
   - `IndexManager`: Manages indexes for collections
   - `query()`: Query with LLM synthesis
   - `retrieve()`: Retrieve nodes without synthesis

#### Models (`src/models/`)

1. **LLM** (`llm.py`): LLM initialization
   - `get_llm()`: Get configured LLM instance
   - Supports Qwen models with quantization options

2. **Embeddings** (`embeddings.py`): Embedding model setup
   - `get_embedding_model()`: Get BGE embedding model

#### Utilities (`src/utils/`)

1. **Config** (`config.py`): Centralized configuration
2. **Logger** (`logger.py`): Centralized logging with debug levels
3. **Faculty ID** (`faculty_id.py`): Faculty ID mapping utilities
4. **Name Matcher** (`name_matcher.py`): Name matching algorithms

### Adding New Data

After adding new CSV rows or PDFs, re-run ingestion:

```bash
# Make sure environment is activated
mamba activate recommender  # or: conda activate recommender

# Re-run ingestion
python scripts/ingest.py
```

**Note:** The ingestion pipeline will:
- Validate CSV format before processing
- Re-index all faculty profiles
- Re-crawl websites if `enable_website_crawling=True` (default)
- Update all ChromaDB collections

### Debugging and Inspection

Use `inspect_nodes.py` to debug data issues:

```bash
# Check what's in the collections
python scripts/inspect_nodes.py --schema

# Inspect specific faculty
python scripts/inspect_nodes.py --query "faculty_name:John Doe"

# Compare collections
python scripts/inspect_nodes.py --collection both --list
```

### Extending the Framework

**Add custom retrieval logic:**
- Modify `FacultyRetriever.retrieve_from_analysis()` in `src/agents/faculty_retriever.py`

**Add custom ranking:**
- Modify `RecommendationAgent.generate_recommendations()` in `src/agents/recommender.py`

**Add new data sources:**
- Create new loader in `src/ingestion/`
- Add to ingestion pipeline in `src/ingestion/pipeline.py`
- Create new ChromaDB collection if needed

**Customize LLM prompts:**
- Modify prompts in `ProposalAnalyzer`, `RecommendationAgent`, or `EnrichmentAgent`

## Troubleshooting

### Out of Memory

- Use a smaller model: `LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct`
- Enable 8-bit quantization (GPU only): `USE_8BIT_QUANTIZATION=true`
- Force CPU mode: `LLM_DEVICE=cpu`
- Close other applications

### Model Download Issues

- Ensure stable internet connection
- Verify HF_TOKEN in .env
- Check disk space (~3-14GB needed depending on model)
- Try again if interrupted

### Slow Inference

- Expected on CPU: 5-15 seconds per generation
- Use GPU for faster inference (~1-3 seconds)
- Consider smaller model if too slow

### ChromaDB Issues

- Delete `faculty_chroma_db/` and re-run ingestion if corrupted
- Ensure sufficient disk space

### GPU Not Detected

- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Verify CUDA installation: `nvidia-smi`
- Set `LLM_DEVICE=cpu` to use CPU mode

## License

[Add your license here]

## Contributors

[Add contributors here]

## Acknowledgments

- Built with LlamaIndex, ChromaDB, and LangChain
- Powered by Qwen2.5-1.5B-Instruct (local) and BGE embeddings
- Web interface built with Streamlit
- Website crawling with requests and BeautifulSoup4

