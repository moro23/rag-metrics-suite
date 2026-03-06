from dataclasses import dataclass
from typing import Dict, Iterable
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    model_name_or_path: str


class Generator:
    """Stub for HF text generation. Not used in the current pipeline."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        logger.info(f"Generator stub initialized for {config.model_name_or_path}")

    def generate(self, data: Iterable[Dict]) -> Iterable[Dict]:
        """Yields input dicts with a placeholder 'answer' field appended."""
        for example in data:
            example["answer"] = f"[placeholder] {example.get('question', '')}"
            yield example