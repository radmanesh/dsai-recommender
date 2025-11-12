"""Help users select the appropriate Qwen model for their hardware."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import psutil


def check_system_resources():
    """Check available system resources and recommend model."""
    print("=" * 70)
    print(" SYSTEM RESOURCES CHECK")
    print("=" * 70)

    # GPU Check
    gpu_available = torch.cuda.is_available()

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✓ GPU Available: {gpu_name}")
        print(f"  VRAM: {gpu_memory:.1f} GB")
    else:
        print(f"\n✗ No GPU detected (will use CPU)")
        try:
            if torch.backends.mps.is_available():
                print("  Note: MPS (Apple Silicon) detected")
        except:
            pass

    # RAM Check
    ram = psutil.virtual_memory().total / 1e9
    ram_available = psutil.virtual_memory().available / 1e9
    print(f"\n  Total RAM: {ram:.1f} GB")
    print(f"  Available RAM: {ram_available:.1f} GB")

    # CPU Check
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    print(f"  CPU Cores: {cpu_count} physical, {cpu_count_logical} logical")

    print("\n" + "=" * 70)
    print(" MODEL RECOMMENDATIONS")
    print("=" * 70)

    if gpu_available:
        recommend_gpu_model(gpu_memory)
    else:
        recommend_cpu_model(ram_available)

    print("\n" + "=" * 70)


def recommend_gpu_model(vram_gb):
    """Recommend model based on GPU memory."""
    print("\n🎮 GPU Mode Recommendations:\n")

    if vram_gb >= 16:
        print("✅ BEST: Qwen/Qwen2.5-Coder-7B-Instruct")
        print("   • Highest quality")
        print("   • ~8GB VRAM required")
        print("   • Inference: ~1-2 seconds")
        print()
        print("✅ GOOD: Qwen/Qwen2.5-Coder-3B-Instruct")
        print("   • Good quality, faster")
        print("   • ~4GB VRAM required")
        print()
        print("✅ FAST: Qwen/Qwen2.5-Coder-1.5B-Instruct (current default)")
        print("   • Fastest inference")
        print("   • ~2GB VRAM required")
    elif vram_gb >= 8:
        print("✅ RECOMMENDED: Qwen/Qwen2.5-Coder-3B-Instruct")
        print("   • Best balance for your GPU")
        print("   • ~4GB VRAM required")
        print("   • Inference: ~2-3 seconds")
        print()
        print("✅ FASTER: Qwen/Qwen2.5-Coder-1.5B-Instruct (current default)")
        print("   • Faster inference")
        print("   • ~2GB VRAM required")
    elif vram_gb >= 4:
        print("✅ RECOMMENDED: Qwen/Qwen2.5-Coder-1.5B-Instruct (current default)")
        print("   • Best fit for your GPU")
        print("   • ~2GB VRAM required")
        print("   • Inference: ~3-5 seconds")
    else:
        print("⚠️  Limited VRAM detected")
        print("✅ RECOMMENDED: Qwen/Qwen2.5-Coder-1.5B-Instruct (current default)")
        print("   • Smallest model")
        print("   • Consider using CPU mode if GPU memory is an issue")


def recommend_cpu_model(ram_available_gb):
    """Recommend model for CPU-only systems."""
    print("\n💻 CPU Mode Recommendations:\n")

    if ram_available_gb >= 16:
        print("✅ BEST: Qwen/Qwen2.5-Coder-3B-Instruct")
        print("   • Good quality on CPU")
        print("   • ~6GB RAM required")
        print("   • Inference: ~10-20 seconds")
        print()
        print("✅ FASTER: Qwen/Qwen2.5-Coder-1.5B-Instruct (current default)")
        print("   • Best for CPU")
        print("   • ~3GB RAM required")
        print("   • Inference: ~5-15 seconds")
    elif ram_available_gb >= 8:
        print("✅ RECOMMENDED: Qwen/Qwen2.5-Coder-1.5B-Instruct (current default)")
        print("   • Ideal for CPU")
        print("   • ~3GB RAM required")
        print("   • Inference: ~5-15 seconds")
        print()
        print("💡 Tip: Close other applications for better performance")
    else:
        print("⚠️  Limited RAM available")
        print("✅ USE: Qwen/Qwen2.5-Coder-1.5B-Instruct (current default)")
        print("   • Smallest model")
        print("   • May be slow with limited RAM")
        print()
        print("💡 Tip: Close other applications before running")


def print_usage_instructions():
    """Print instructions for changing the model."""
    print("\n" + "=" * 70)
    print(" HOW TO CHANGE MODEL")
    print("=" * 70)
    print("""
To use a different model, update your .env file:

1. Open or create .env in the project root
2. Set the LLM_MODEL variable:

   LLM_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct  # Current default (CPU-friendly)
   # or
   LLM_MODEL=Qwen/Qwen2.5-Coder-3B-Instruct    # Medium (needs GPU or good CPU)
   # or
   LLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct    # Large (needs GPU with 8GB+ VRAM)

3. First run will download the model (one-time, ~3-14GB depending on model)
4. Subsequent runs will use the cached model

Optional: Enable 8-bit quantization (reduces memory by ~50%, GPU only):
   USE_8BIT_QUANTIZATION=true

Model cache location: ~/.cache/huggingface/hub/
""")


def main():
    """Main function."""
    print("\n" + "=" * 70)
    print(" QWEN MODEL SELECTOR FOR FACULTY MATCHMAKER")
    print("=" * 70)

    try:
        check_system_resources()
        print_usage_instructions()

        print("\n" + "=" * 70)
        print(" CURRENT CONFIGURATION")
        print("=" * 70)

        try:
            from src.utils.config import Config
            print(f"\nCurrent model: {Config.LLM_MODEL}")
            print(f"8-bit quantization: {Config.USE_8BIT_QUANTIZATION}")
            print(f"Device: {Config.LLM_DEVICE}")
        except Exception as e:
            print(f"\nCould not load config: {e}")
            print("Make sure .env is configured properly")

        print("\n✅ Ready to proceed!")
        print("\nNext steps:")
        print("  1. Update .env if needed (see instructions above)")
        print("  2. Run: python scripts/test_local_model.py")
        print("  3. Then: python scripts/ingest.py")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

