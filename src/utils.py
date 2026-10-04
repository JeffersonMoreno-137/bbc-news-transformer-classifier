import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def plot_training_history(
    history_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Plot and optionally persist training and validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Validation")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Accuracy curve
    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], marker="o", label="Train")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], marker="o", label="Validation")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
) -> None:
    """Render and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            val = cm[row, col]
            color = "white" if val > (cm.max() / 2) else "black"
            ax.text(col, row, str(val), ha="center", va="center", color=color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
