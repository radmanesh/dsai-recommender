# Agentic Research Matchmaker

An intelligent recommendation system that matches Ph.D. proposals to OU faculty using semantic search and multi-agent orchestration.

## Overview

This system uses:
- **LlamaIndex** for document processing and indexing
- **ChromaDB** for vector storage
- **Qwen2.5-Coder-1.5B-Instruct** for reasoning (runs locally)
- **HuggingFace BGE embeddings** (BAAI/bge-small-en-v1.5) for semantic search
- **LangChain** for multi-agent orchestration

## Features

- 📄 Ingest faculty data from CSV and PDFs (CVs, papers, proposals)
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
├── data/
│   ├── DSAI-Faculties.csv          # Faculty metadata
│   └── pdfs/                        # Faculty CVs, papers, proposals
├── src/
│   ├── ingestion/                   # Data loading and processing
│   ├── indexing/                    # Vector store and index management
│   ├── agents/                      # Multi-agent implementation
│   ├── models/                      # LLM and embedding setup
│   └── utils/                       # Configuration and utilities
├── scripts/
│   ├── ingest.py                    # Data ingestion script
│   └── query_demo.py                # Query demonstration
└── requirements.txt                 # Python dependencies
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
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

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
LLM_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct  # Default, runs on CPU
```

Get your token from: https://huggingface.co/settings/tokens

**Available Models:**
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` - Best for CPU (~3GB)
- `Qwen/Qwen2.5-Coder-3B-Instruct` - Medium, needs GPU/good CPU (~6GB)
- `Qwen/Qwen2.5-Coder-7B-Instruct` - Large, needs GPU 8GB+ VRAM (~14GB)

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

Run the ingestion pipeline to index all data:

```bash
python scripts/ingest.py
```

This will:
- Parse CSV and PDF files
- Create embeddings using BGE
- Store vectors in ChromaDB

## Usage

### Web Interface (Recommended)

Launch the user-friendly web interface:

```bash
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

For scripting and automation:

```bash
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
- `LLM_MODEL`: Local LLM model to use (default: Qwen2.5-Coder-1.5B-Instruct)
- `USE_8BIT_QUANTIZATION`: Reduce memory usage by 50% (GPU only, default: false)
- `LLM_DEVICE`: Device to use (auto/cuda/cpu, default: auto)
- `CHROMA_PATH`: ChromaDB storage path
- `COLLECTION_NAME`: ChromaDB collection name
- `EMBEDDING_MODEL`: Embedding model name
- `TOP_K_RESULTS`: Number of results to retrieve
- `CHUNK_SIZE`: Text chunk size for splitting
- `CHUNK_OVERLAP`: Overlap between chunks

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
python scripts/ingest.py
```

## Troubleshooting

### Out of Memory

- Use a smaller model: `LLM_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct`
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
- Powered by Qwen2.5-Coder-32B-Instruct and BGE embeddings

