from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import BEST_MODEL_PATH, LEARNING_RATE, NUM_EPOCHS


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model over single epoch, returning average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    epoch_loss = total_loss / total_examples
    epoch_acc = correct_predictions / total_examples
    return epoch_loss, epoch_acc


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model over single epoch without gradient calculation."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    epoch_loss = total_loss / total_examples
    epoch_acc = correct_predictions / total_examples
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    checkpoint_path: Path = BEST_MODEL_PATH,
) -> pd.DataFrame:
    """Execute complete fine-tuning routine, retaining checkpoint with minimum validation loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")

    print(f"Beginning training across {epochs} epochs on device: {device}")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: val_loss improved to {val_loss:.4f} -> {checkpoint_path}")

    history_df = pd.DataFrame(history)
    return history_df
