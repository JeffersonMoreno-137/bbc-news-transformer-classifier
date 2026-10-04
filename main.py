import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from src.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DEVICE,
    FIGURES_DIR,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_EPOCHS,
    RANDOM_SEED,
)
from src.data import (
    create_stratified_splits,
    get_dataloaders,
    load_raw_dataset,
    preprocess_data,
)
from src.evaluate import evaluate_test_set, predict_single_text
from src.model import build_model, count_parameters, load_tokenizer
from src.train import train_model
from src.utils import plot_confusion_matrix, plot_training_history, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BBC News Transformer Classification Pipeline (BERT fine-tuning in PyTorch)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional local path to bbc_text_cls.csv. If omitted, attempts local discovery or KaggleHub download.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of fine-tuning epochs (default: {NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for DataLoader (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate for AdamW optimizer (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help=f"Max token sequence length (default: {MAX_LENGTH}).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "predict"],
        default="evaluate",
        help="Operation mode: 'train' (full train + eval), 'evaluate' (eval checkpoint), or 'predict'.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The company reported higher quarterly profits after strong sales in international markets.",
        help="Input text for single prediction when mode='predict'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)

    print("=================================================================")
    print("      BBC News Transformer Multi-Class Classifier (BERT)        ")
    print("=================================================================")
    print(f"Device: {DEVICE}")
    print(f"Mode:   {args.mode}")

    tokenizer = load_tokenizer(MODEL_NAME)

    if args.mode in ["train", "evaluate"]:
        print("\n[1/4] Loading and preprocessing dataset...")
        df_raw = load_raw_dataset(args.data_path)
        df_clean, encoder, id2label, label2id = preprocess_data(df_raw)
        train_df, val_df, test_df = create_stratified_splits(df_clean, RANDOM_SEED)

        print(f"Total dataset: {len(df_raw)} rows -> {len(df_clean)} cleaned rows")
        print(f"Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")

        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df, tokenizer, args.batch_size, args.max_length
        )

        model = build_model(
            num_labels=len(encoder.classes_),
            id2label=id2label,
            label2id=label2id,
            model_name=MODEL_NAME,
            device=DEVICE,
        )
        total_p, trainable_p = count_parameters(model)
        print(f"BERT parameters: Total = {total_p:,}, Trainable = {trainable_p:,}")

        if args.mode == "train":
            print("\n[2/4] Commencing fine-tuning...")
            history_df = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=DEVICE,
                epochs=args.epochs,
                learning_rate=args.lr,
                checkpoint_path=BEST_MODEL_PATH,
            )
            plot_training_history(history_df, FIGURES_DIR / "03_training_validation_curves.png")
            print("Saved training curves to figures/03_training_validation_curves.png")

        # Load best checkpoint for testing
        if BEST_MODEL_PATH.exists():
            print(f"\n[3/4] Loading best model checkpoint from {BEST_MODEL_PATH}...")
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        else:
            print(f"\nNotice: {BEST_MODEL_PATH} not found. Using current model weights.")

        print("\n[4/4] Evaluating on hold-out Test Set...")
        eval_results = evaluate_test_set(model, test_loader, list(encoder.classes_), DEVICE)
        print(f"\nFinal Test Accuracy: {eval_results['accuracy']:.4f}\n")
        print("Classification Report:")
        print(eval_results["classification_report_text"])

        plot_confusion_matrix(
            eval_results["confusion_matrix"],
            list(encoder.classes_),
            FIGURES_DIR / "04_confusion_matrix.png",
        )
        print("Updated confusion matrix plot in figures/04_confusion_matrix.png")

    elif args.mode == "predict":
        # Predict mode: construct label map from known BBC categories
        classes = ["business", "entertainment", "politics", "sport", "tech"]
        id2label = {i: c for i, c in enumerate(classes)}
        label2id = {c: i for i, c in enumerate(classes)}

        model = build_model(
            num_labels=len(classes),
            id2label=id2label,
            label2id=label2id,
            model_name=MODEL_NAME,
            device=DEVICE,
        )
        if BEST_MODEL_PATH.exists():
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
            print(f"Loaded weights from {BEST_MODEL_PATH}")
        else:
            print("Warning: Running prediction with base initialized weights (checkpoint not found).")

        result = predict_single_text(model, tokenizer, args.text, id2label, DEVICE, args.max_length)
        print("\nInference Output:")
        print(f"Input Text:       \"{result['text']}\"")
        print(f"Predicted Class:  {result['predicted_label'].upper()}")
        print(f"Confidence:       {result['confidence']:.4f}")
        print("\nProbability Distribution:")
        for label, prob in result["all_probabilities"].items():
            print(f"  - {label.ljust(15)}: {prob:.4f}")


if __name__ == "__main__":
    main()
