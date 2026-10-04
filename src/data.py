from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from src.config import (
    BATCH_SIZE,
    DATASET_DIR,
    KAGGLE_DATASET_HANDLE,
    LABEL_COLUMN,
    MAX_LENGTH,
    RANDOM_SEED,
    TEST_SPLIT_SIZE,
    TEXT_COLUMN,
    VAL_TEST_SPLIT_RATIO,
)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove excessive carriage returns."""
    return " ".join(str(text).split())


def load_raw_dataset(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load dataset from provided path or download automatically from Kaggle.
    """
    if csv_path and Path(csv_path).exists():
        target_file = Path(csv_path)
    else:
        # Check local dataset directory
        potential_files = list(DATASET_DIR.glob("**/*.csv"))
        if potential_files:
            target_file = potential_files[0]
        else:
            try:
                import kagglehub
                DATASET_DIR.mkdir(parents=True, exist_ok=True)
                download_path = Path(
                    kagglehub.dataset_download(KAGGLE_DATASET_HANDLE, output_dir=DATASET_DIR)
                )
                csv_files = list(download_path.glob("**/*.csv"))
                if not csv_files:
                    raise FileNotFoundError(f"No CSV found in downloaded dataset at {download_path}")
                target_file = csv_files[0]
            except Exception as e:
                raise RuntimeError(
                    f"Unable to load or download dataset: {e}. "
                    "Ensure kagglehub is installed or provide a valid CSV path."
                )

    df = pd.read_csv(target_file)
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Expected columns '{TEXT_COLUMN}' and '{LABEL_COLUMN}' in dataset. Found: {list(df.columns)}"
        )
    return df


def preprocess_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, LabelEncoder, Dict[int, str], Dict[str, int]]:
    """Clean text, drop duplicates, and encode labels numerically."""
    df_clean = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    df_clean[TEXT_COLUMN] = df_clean[TEXT_COLUMN].apply(clean_text)
    df_clean = df_clean.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)

    encoder = LabelEncoder()
    df_clean["label_id"] = encoder.fit_transform(df_clean[LABEL_COLUMN])

    id2label = {idx: label for idx, label in enumerate(encoder.classes_)}
    label2id = {label: idx for idx, label in id2label.items()}

    return df_clean, encoder, id2label, label2id


def create_stratified_splits(
    df: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into 80% train, 10% validation, and 10% test with stratified labels."""
    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SPLIT_SIZE,
        stratify=df["label_id"],
        random_state=random_seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=VAL_TEST_SPLIT_RATIO,
        stratify=temp_df["label_id"],
        random_state=random_seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class BBCNewsDataset(Dataset):
    """PyTorch Dataset wrapper for tokenized BBC news text."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.texts = dataframe[TEXT_COLUMN].tolist()
        self.labels = dataframe["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test PyTorch DataLoaders."""
    train_ds = BBCNewsDataset(train_df, tokenizer, max_length)
    val_ds = BBCNewsDataset(val_df, tokenizer, max_length)
    test_ds = BBCNewsDataset(test_df, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
