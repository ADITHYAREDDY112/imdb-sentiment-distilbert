"""Run sentiment inference with the fine-tuned IMDb model."""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict IMDb sentiment for text.")
    parser.add_argument("--model-dir", default="models/imdb-distilbert")
    parser.add_argument(
        "--text",
        required=True,
        help="Raw movie review text to classify.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def predict_sentiment(text: str, model_dir: str = "models/imdb-distilbert", max_length: int = 256):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1).squeeze()

    predicted_id = int(torch.argmax(probabilities).item())
    label = model.config.id2label.get(predicted_id, str(predicted_id)).lower()
    confidence = float(probabilities[predicted_id].item())

    return {
        "text": text,
        "label": label,
        "confidence": confidence,
        "probabilities": {
            model.config.id2label.get(index, str(index)).lower(): float(score.item())
            for index, score in enumerate(probabilities)
        },
    }


def main() -> None:
    args = parse_args()
    prediction = predict_sentiment(args.text, args.model_dir, args.max_length)
    print(json.dumps(prediction, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
