from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from src.config import MAX_LENGTH
from src.data import clean_text


def predict_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform batched inference over entire dataloader."""
    model.eval()
    true_labels: List[int] = []
    predicted_labels: List[int] = []
    predicted_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())
            predicted_probs.extend(probabilities.cpu().numpy())

    return (
        np.array(true_labels),
        np.array(predicted_labels),
        np.array(predicted_probs),
    )


def evaluate_test_set(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Dict[str, object]:
    """Compute overall accuracy, detailed classification metrics, and confusion matrix."""
    y_true, y_pred, y_prob = predict_dataset(model, dataloader, device)
    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def predict_single_text(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    text: str,
    id2label: Dict[int, str],
    device: torch.device,
    max_length: int = MAX_LENGTH,
) -> Dict[str, object]:
    """Run inference on arbitrary input text with class confidence score."""
    model.eval()
    cleaned = clean_text(text)
    encoded = tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1).squeeze(0)
        predicted_id = probabilities.argmax().item()

    return {
        "text": text,
        "predicted_label": id2label[predicted_id],
        "confidence": float(probabilities[predicted_id].item()),
        "all_probabilities": {
            id2label[i]: float(prob) for i, prob in enumerate(probabilities.cpu().numpy())
        },
    }
