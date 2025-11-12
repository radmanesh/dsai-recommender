# Quick Start Guide

Get up and running with the Agentic Research Matchmaker in 10 minutes.

## Prerequisites

- Python 3.8 or higher
- HuggingFace account and API token
- **5GB+ free disk space** (for local model)
- **8GB+ RAM** (16GB recommended)
- GPU optional but recommended for faster inference

## Installation

### 1. Clone and Setup Environment

```bash
cd dsai-recommender
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Check Your Hardware

```bash
python scripts/select_model.py
```

This shows your system resources and recommends the best Qwen model size for your hardware.

### 3. Configure Environment Variables

```bash
cp .env.template .env
```

Edit `.env` and configure:

```bash
HF_TOKEN=your_actual_token_here
LLM_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct  # Default (CPU-friendly)
```

**Model Options:**
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` - Smallest, runs on CPU (~3GB)
- `Qwen/Qwen2.5-Coder-3B-Instruct` - Medium, needs GPU or good CPU (~6GB)
- `Qwen/Qwen2.5-Coder-7B-Instruct` - Largest, needs GPU with 8GB+ VRAM (~14GB)

Get your token from: https://huggingface.co/settings/tokens

### 4. Test Local Model

```bash
python scripts/test_local_model.py
```

**First run:** Downloads the model (~3GB, one-time, takes 2-5 minutes)
**Subsequent runs:** Uses cached model (loads in 30-60 seconds)

## First Run

### Step 5: Prepare Data (Already Done!)

Sample data is included:
- ✓ `data/DSAI-Faculties.csv` - 10+ sample faculty profiles
- ✓ `data/sample_proposal.pdf` - Example PhD proposal
- ✓ `data/pdfs/` - Faculty profile PDFs

### Step 6: Index the Data

```bash
python scripts/ingest.py
```

This will:
- Load faculty CSV
- Load PDFs from `data/pdfs/` (if any)
- Create embeddings
- Store in ChromaDB

Expected output:
```
✅ Ingestion completed successfully!
Created XX nodes in the vector store
```

### Step 3: Run Demo

```bash
python scripts/demo.py
```

This demonstrates:
- Text-based proposal matching
- Quick search functionality
- PDF proposal matching (if available)

## Try It Yourself

### Interactive Query Mode

```bash
python scripts/query_demo.py --interactive
```

Try queries like:
- "multi-agent systems and reinforcement learning"
- "code generation with large language models"
- "natural language processing"

### Programmatic Usage

```python
from src.agents.orchestrator import ResearchMatchOrchestrator

# Initialize
orchestrator = ResearchMatchOrchestrator(top_n_recommendations=5)

# Analyze a proposal
result = orchestrator.match_proposal_text("""
    Research on multi-agent reinforcement learning
    for cooperative AI systems...
""")

# View recommendations
for rec in result.recommendations:
    print(f"{rec.rank}. {rec.faculty_name}")
    print(f"   {rec.explanation}\n")
```

### Single Query

```bash
python scripts/query_demo.py --query "natural language processing"
```

## Add Your Own Data

### Add Faculty

Edit `data/DSAI-Faculties.csv` and add rows:

```csv
name,role,department,areas,research_interests,email,website
Dr. New Faculty,Professor,CS,AI; ML,"Research description...",email@ou.edu,https://...
```

### Add PDFs

Place PDF files in `data/pdfs/`:

```bash
data/pdfs/
├── DrSmith_CV.pdf
├── DrSmith_paper_neural_nets.pdf
└── sample_proposal_robotics.pdf
```

### Re-index

After adding data:

```bash
python scripts/ingest.py
```

Choose "Reset collection" if you want to start fresh.

## Common Issues

### "Out of memory" / "CUDA out of memory"

**Solution**:
- Use smaller model: `LLM_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Enable 8-bit quantization (GPU only): `USE_8BIT_QUANTIZATION=true`
- Force CPU mode: `LLM_DEVICE=cpu`
- Close other applications

### "HF_TOKEN is required"

**Solution**: Set `HF_TOKEN` in your `.env` file (needed for downloading model)

### "Collection is empty"

**Solution**: Run `python scripts/ingest.py` to index data

### "Model download slow"

**Solution**: First run downloads model (~3GB for 1.5B). This is normal and one-time.
- Takes 2-5 minutes depending on internet speed
- Model cached at `~/.cache/huggingface/hub/`

### "Inference is slow"

**Solution**:
- CPU mode: 5-15 seconds per generation is normal
- Use GPU for faster inference (1-3 seconds)
- Consider lighter model if too slow

### Import errors

**Solution**: Ensure you activated the virtual environment:
```bash
source venv/bin/activate
```

## Next Steps

1. **Add your real data**: Replace sample CSV and add real PDFs
2. **Customize configuration**: Edit `.env` to tune parameters
3. **Build UI**: Integrate with Streamlit or Flask for web interface
4. **Deploy agents**: Use Google Agent Development Kit for production

## Getting Help

- Check `README.md` for detailed documentation
- Run `python scripts/test_system.py` to diagnose issues
- Review logs in the terminal output

## Architecture Overview

```
User Input (PDF/Text)
        ↓
[Proposal Analyzer Agent]
        ↓
[Faculty Retrieval Agent] ←→ ChromaDB Vector Store
        ↓
[Recommendation Agent]
        ↓
Ranked Recommendations + Explanations
```

Each agent uses:
- **Embeddings**: BAAI/bge-small-en-v1.5 (local, CPU)
- **LLM**: Qwen/Qwen2.5-Coder-1.5B-Instruct (local, CPU/GPU)
- **Vector DB**: ChromaDB (local)

## Performance Tips

- **Model selection**: Use 1.5B for CPU, 3B/7B for GPU
- **Quantization**: Enable 8-bit on GPU to reduce memory by 50%
- **Chunk size**: Larger chunks (1024+) for academic text
- **Top K**: Start with 10, increase for more comprehensive results
- **Temperature**: Lower (0.1-0.3) for consistent outputs
- **Batch operations**: Process multiple queries together for efficiency

Enjoy using the Agentic Research Matchmaker! 🎉

