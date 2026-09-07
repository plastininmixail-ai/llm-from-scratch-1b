"""Framework inference package."""
from .generator import generate, load_model_for_inference, load_tokenizer

__all__ = ["generate", "load_model_for_inference", "load_tokenizer"]
