"""Framework top-level package."""
from .model import GPT, ModelConfig, load_config, count_params
from .data import BinaryDataset, tokenize_jsonl
from .training import Trainer, TrainerConfig
from .inference import generate, load_model_for_inference
from .evaluation import compute_perplexity, evaluate_checkpoint

__all__ = [
    "GPT", "ModelConfig", "load_config", "count_params",
    "BinaryDataset", "tokenize_jsonl",
    "Trainer", "TrainerConfig",
    "generate", "load_model_for_inference",
    "compute_perplexity", "evaluate_checkpoint",
]
