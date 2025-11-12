"""Test local model loading and inference."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def test_model_loading():
    """Test that model loads successfully."""
    print("=" * 70)
    print(" LOCAL MODEL TEST")
    print("=" * 70)

    # Show system info
    print("\nSystem Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    # Show configuration
    try:
        from src.utils.config import Config
        print("Current Configuration:")
        print(f"  Model: {Config.LLM_MODEL}")
        print(f"  Temperature: {Config.LLM_TEMPERATURE}")
        print(f"  Max tokens: {Config.LLM_MAX_TOKENS}")
        print(f"  8-bit quantization: {Config.USE_8BIT_QUANTIZATION}")
        print()
    except Exception as e:
        print(f"⚠️  Could not load config: {e}")
        print("Make sure .env is configured\n")
        return False

    # Test model loading
    print("=" * 70)
    print("Loading model (this may take a few minutes on first run)...")
    print("=" * 70)
    print()

    try:
        from src.models.llm import get_llm

        llm = get_llm()
        print("\n✓ Model loaded successfully!")

        # Test inference
        print("\n" + "=" * 70)
        print("Testing inference...")
        print("=" * 70)

        test_prompt = "Explain what machine learning is in one sentence."
        print(f"\nPrompt: {test_prompt}")
        print("\nGenerating response...")

        response = llm.complete(test_prompt)

        print("\n" + "-" * 70)
        print("Response:")
        print("-" * 70)
        print(str(response))
        print("-" * 70)

        print("\n✓ Inference successful!")

        # Test a second query (should be faster with cached model)
        print("\n" + "=" * 70)
        print("Testing second inference (should be faster)...")
        print("=" * 70)

        test_prompt_2 = "What is natural language processing?"
        print(f"\nPrompt: {test_prompt_2}")
        print("Generating response...")

        response_2 = llm.complete(test_prompt_2)

        print("\n" + "-" * 70)
        print("Response:")
        print("-" * 70)
        print(str(response_2))
        print("-" * 70)

        print("\n✓ Second inference successful!")

        print("\n" + "=" * 70)
        print(" TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print()
        print("✅ Local model is working correctly!")
        print()
        print("Next steps:")
        print("  1. Run: python scripts/ingest.py")
        print("  2. Try: python match.py 'machine learning research'")
        print("  3. Or: python scripts/demo.py")
        print()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print()
        import traceback
        traceback.print_exc()

        print("\n" + "=" * 70)
        print(" TROUBLESHOOTING")
        print("=" * 70)
        print("""
Common issues:

1. Out of Memory:
   - Try a smaller model in .env:
     LLM_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
   - Enable 8-bit quantization (GPU only):
     USE_8BIT_QUANTIZATION=true
   - Close other applications

2. CUDA errors:
   - Make sure PyTorch is installed with CUDA support
   - Try CPU mode:
     LLM_DEVICE=cpu

3. Download issues:
   - Check internet connection
   - Verify HF_TOKEN in .env (needed for download)
   - Try again (downloads can be interrupted)

4. Import errors:
   - Install dependencies:
     pip install -r requirements.txt

For more help, run: python scripts/select_model.py
""")
        return False


def main():
    """Main function."""
    success = test_model_loading()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

