"""Evaluate a saved IMDb sentiment model and write metrics/plots to results/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABEL_NAMES = ["negative", "positive"]


def import_hf_evaluate():
    """Import Hugging Face evaluate without being shadowed by this script."""
    script_dir = Path(__file__).resolve().parent
    original_path = list(sys.path)
    sys.path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() != script_dir
    ]
    try:
        import evaluate as hf_evaluate
    finally:
        sys.path = original_path
    return hf_evaluate


hf_evaluate = import_hf_evaluate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned IMDb model.")
    parser.add_argument("--model-dir", default="models/imdb-distilbert")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_subset(dataset, sample_count: int, seed: int):
    if sample_count < 0 or sample_count >= len(dataset):
        return dataset.shuffle(seed=seed)
    return dataset.shuffle(seed=seed).select(range(sample_count))


def build_compute_metrics():
    accuracy_metric = hf_evaluate.load("accuracy")
    precision_metric = hf_evaluate.load("precision")
    recall_metric = hf_evaluate.load("recall")
    f1_metric = hf_evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(
                predictions=predictions, references=labels
            )["accuracy"],
            "precision": precision_metric.compute(
                predictions=predictions, references=labels, average="binary"
            )["precision"],
            "recall": recall_metric.compute(
                predictions=predictions, references=labels, average="binary"
            )["recall"],
            "f1": f1_metric.compute(
                predictions=predictions, references=labels, average="binary"
            )["f1"],
        }

    return compute_metrics


def normalize_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    normalized = {}
    for key in ("accuracy", "precision", "recall", "f1"):
        value = metrics.get(f"{prefix}_{key}", metrics.get(key))
        if value is not None:
            normalized[key] = float(value)
    return normalized


def plot_confusion_matrix(labels, predictions, output_path: Path) -> None:
    matrix = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("IMDb Sentiment Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    raw_test = load_dataset("stanfordnlp/imdb", split="test")
    raw_test = select_subset(raw_test, args.test_samples, args.seed)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized_test = raw_test.map(tokenize_batch, batched=True).remove_columns(["text"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="checkpoints/eval",
            per_device_eval_batch_size=args.batch_size,
            report_to="none",
            seed=args.seed,
        ),
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(),
    )

    raw_metrics = trainer.evaluate(metric_key_prefix="test")
    predictions = trainer.predict(tokenized_test)
    predicted_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = np.asarray(predictions.label_ids)

    confusion_matrix_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(true_labels, predicted_labels, confusion_matrix_path)

    metrics = {
        "model_dir": str(model_dir),
        "test_samples": len(tokenized_test),
        "fine_tuned": normalize_metrics(raw_metrics, "test"),
        "confusion_matrix": str(confusion_matrix_path),
    }
    metrics_path = results_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
