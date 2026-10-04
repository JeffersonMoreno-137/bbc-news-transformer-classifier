from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertForSequenceClassification, PreTrainedTokenizer

from src.config import MODEL_NAME


def load_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizer:
    """Load pretrained Hugging Face tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


def build_model(
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    model_name: str = MODEL_NAME,
    device: torch.device = torch.device("cpu"),
) -> BertForSequenceClassification:
    """
    Instantiate BertForSequenceClassification fine-tuning architecture.
    """
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
