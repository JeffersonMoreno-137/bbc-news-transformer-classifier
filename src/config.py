from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models_saved"
FIGURES_DIR = BASE_DIR / "figures"
DOCS_DIR = BASE_DIR / "docs"

# Ensure runtime directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Dataset specifications
KAGGLE_DATASET_HANDLE = "jacopoferretti/bbc-articles-dataset"
TEXT_COLUMN = "text"
LABEL_COLUMN = "labels"
TEST_SPLIT_SIZE = 0.20
VAL_TEST_SPLIT_RATIO = 0.50  # 10% val, 10% test

# Model and Tokenizer settings
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

# Checkpoint paths
BEST_MODEL_PATH = MODELS_DIR / "best_bert_bbc_model.pt"

# Device selection: CUDA -> MPS -> CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
