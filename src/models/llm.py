"""LLM setup and management using local HuggingFace models."""

from llama_index.llms.huggingface import HuggingFaceLLM
import torch
from src.utils.config import Config

# Global LLM instance
_llm = None


def get_llm() -> HuggingFaceLLM:
    """
    Get or create the LLM singleton.

    Returns:
        HuggingFaceLLM: The LLM instance configured with local Qwen model.
    """
    global _llm

    if _llm is None:
        print(f"Loading local LLM: {Config.LLM_MODEL}")

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        # Determine dtype based on device
        if device == "cuda":
            dtype = torch.float16
            print(f"  Using float16 for GPU")
        else:
            dtype = torch.float32
            print(f"  Using float32 for CPU")

        # Build model kwargs
        model_kwargs = {
            "torch_dtype": dtype,
        }

        # Add quantization if enabled and on CUDA
        if Config.USE_8BIT_QUANTIZATION and device == "cuda":
            model_kwargs["load_in_8bit"] = True
            print(f"  Using 8-bit quantization")

        # Create LLM with local model
        _llm = HuggingFaceLLM(
            model_name=Config.LLM_MODEL,
            tokenizer_name=Config.LLM_MODEL,
            device_map="auto",
            model_kwargs=model_kwargs,
            generate_kwargs={
                "temperature": Config.LLM_TEMPERATURE,
                "do_sample": True,
            },
            max_new_tokens=Config.LLM_MAX_TOKENS,  # Pass as top-level parameter
        )

        print("✓ Local LLM loaded successfully")

    return _llm


def reset_llm():
    """Reset the LLM singleton (useful for testing)."""
    global _llm
    _llm = None


def get_llm_with_params(temperature: float = None, max_tokens: int = None) -> HuggingFaceLLM:
    """
    Get a new LLM instance with custom parameters.

    Args:
        temperature: Override default temperature
        max_tokens: Override default max tokens

    Returns:
        HuggingFaceLLM: A new LLM instance with specified parameters.
    """
    temp = temperature if temperature is not None else Config.LLM_TEMPERATURE
    max_tok = max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Build model kwargs
    model_kwargs = {
        "torch_dtype": dtype,
    }

    if Config.USE_8BIT_QUANTIZATION and device == "cuda":
        model_kwargs["load_in_8bit"] = True

    return HuggingFaceLLM(
        model_name=Config.LLM_MODEL,
        tokenizer_name=Config.LLM_MODEL,
        device_map="auto",
        model_kwargs=model_kwargs,
        generate_kwargs={
            "temperature": temp,
            "do_sample": True,
        },
        max_new_tokens=max_tok,  # Pass as top-level parameter
    )
