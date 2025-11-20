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

The system consists of three main agents:

1. **Proposal Analysis Agent**: Extracts key topics, methods, and domains from uploaded proposals
2. **Faculty Retrieval Agent**: Performs semantic search over faculty data using ChromaDB
3. **Recommendation Agent**: Ranks matches and generates explanations using Qwen2.5-Coder

An orchestrator coordinates the workflow: upload → analysis → retrieval → recommendation.

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
- `data/DSAI-Faculties.csv` - Faculty metadata (name, role, department, areas, research interests, etc.)
- `data/pdfs/` - Faculty CVs, publications, sample proposals, etc.

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

This will:
- Parse CSV and PDF files
- Crawl faculty websites and lab websites (if URLs are provided in CSV)
- Create embeddings using BGE
- Store vectors in ChromaDB collections:
  - `faculty_profiles` - Faculty metadata from CSV
  - `faculty_pdfs` - PDF documents (CVs, papers, etc.)
  - `faculty_websites` - Crawled website content

## Usage

### Web Interface (Recommended)

**Important:** Make sure your conda/mamba environment is activated before running:

```bash
# Activate environment first
mamba activate recommender  # or: conda activate recommender

# Method 1: Direct
streamlit run app.py

# Method 2: Via launcher script
python scripts/run_app.py
```

Access at: **http://localhost:8501**

**Features:**
- 📄 **Drag-and-drop PDF upload** - Upload proposal PDFs easily
- ✍️ **Text input** - Paste proposal text directly
- 🔍 **Quick search** - Search by keywords or research areas
- 📊 **Interactive results** - Expandable faculty cards with details
- ✉️ **Email drafts** - Auto-generate contact emails

### Command-Line Interface

**Important:** Make sure your conda/mamba environment is activated:

```bash
# Activate environment first
mamba activate recommender  # or: conda activate recommender

# Command-line interface
python scripts/match.py "Your proposal text here..."
python scripts/match.py --file proposal.pdf
python scripts/match.py --query "machine learning"
```

### Query Demo

Test the system with example queries:

```bash
python scripts/query_demo.py --interactive
```

### Programmatic Usage

Use in your Python code:

```python
from src.agents.orchestrator import ResearchMatchOrchestrator

# Initialize orchestrator
orchestrator = ResearchMatchOrchestrator()

# Analyze a proposal and get recommendations
recommendations = orchestrator.match_proposal("path/to/proposal.pdf")

# Display results
for rec in recommendations:
    print(f"Faculty: {rec['name']}")
    print(f"Score: {rec['score']}")
    print(f"Reason: {rec['explanation']}")
    print("---")
```

## Data Format

### CSV Format (DSAI-Faculties.csv)

Expected columns:
- `name`: Faculty name
- `role`: Position (Professor, Associate Professor, etc.)
- `department`: Department name
- `areas`: Research areas (comma-separated)
- `research_interests`: Detailed research interests
- `website`: Faculty website URL
- `email`: Contact email

### PDF Files

PDFs should be organized in `data/pdfs/` with clear naming:
- `{faculty_name}_CV.pdf`
- `{faculty_name}_paper_{title}.pdf`
- `sample_proposal_{topic}.pdf`

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

