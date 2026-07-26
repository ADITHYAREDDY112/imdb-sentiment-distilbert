"""Fine-tune DistilBERT on IMDb sentiment classification.

The defaults intentionally use a moderate subset so the project can run on a
laptop. Increase --train-samples/--eval-samples/--test-samples, or pass -1, for
larger experiments.
"""

from __future__ import annotations

import argparse
import json
import inspect
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import DatasetDict, load_dataset
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
    """Import Hugging Face evaluate without being shadowed by local evaluate.py."""
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
    parser = argparse.ArgumentParser(description="Fine-tune a transformer on IMDb.")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="models/imdb-distilbert")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--eval-samples", type=int, default=3000)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip evaluating the pretrained encoder with an untrained classifier head.",
    )
    return parser.parse_args()


def select_subset(dataset, sample_count: int, seed: int):
    if sample_count < 0 or sample_count >= len(dataset):
        return dataset.shuffle(seed=seed)
    return dataset.shuffle(seed=seed).select(range(sample_count))


def load_imdb_splits(args: argparse.Namespace) -> DatasetDict:
    imdb = load_dataset("imdb")
    train_validation = imdb["train"].train_test_split(test_size=0.2, seed=args.seed)

    return DatasetDict(
        {
            "train": select_subset(train_validation["train"], args.train_samples, args.seed),
            "validation": select_subset(
                train_validation["test"], args.eval_samples, args.seed
            ),
            "test": select_subset(imdb["test"], args.test_samples, args.seed),
        }
    )


def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_batch, batched=True)
    return tokenized.remove_columns(["text"])


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


def save_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    common_args = {
        "output_dir": "checkpoints/imdb-distilbert",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "report_to": "none",
        "seed": args.seed,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        common_args["eval_strategy"] = "epoch"
    else:
        common_args["evaluation_strategy"] = "epoch"
    return TrainingArguments(**common_args)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    raw_dataset = load_imdb_splits(args)
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = build_compute_metrics()

    training_args = build_training_arguments(args)

    metrics: dict[str, Any] = {
        "model_name": args.model_name,
        "train_samples": len(tokenized_dataset["train"]),
        "validation_samples": len(tokenized_dataset["validation"]),
        "test_samples": len(tokenized_dataset["test"]),
    }

    if not args.skip_baseline:
        baseline_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2
        )
        baseline_trainer = Trainer(
            model=baseline_model,
            args=training_args,
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        baseline_metrics = baseline_trainer.evaluate(
            eval_dataset=tokenized_dataset["test"], metric_key_prefix="baseline"
        )
        metrics["baseline"] = normalize_metrics(baseline_metrics, "baseline")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["test"], metric_key_prefix="test"
    )
    predictions = trainer.predict(tokenized_dataset["test"])
    predicted_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = np.asarray(predictions.label_ids)

    confusion_matrix_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(true_labels, predicted_labels, confusion_matrix_path)

    metrics["fine_tuned"] = normalize_metrics(test_metrics, "test")
    metrics["confusion_matrix"] = str(confusion_matrix_path)
    save_json(metrics, results_dir / "metrics.json")

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
