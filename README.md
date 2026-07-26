# IMDb Sentiment Classification with DistilBERT

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![DistilBERT](https://img.shields.io/badge/Model-DistilBERT-5b6ee1)

Fine-tune a pretrained transformer for binary movie-review sentiment classification. The project uses Hugging Face `transformers`, `datasets`, and `evaluate` with PyTorch and the `Trainer` API to train, evaluate, save, and run inference with a DistilBERT classifier.

## Dataset

This project uses the [`imdb`](https://huggingface.co/datasets/imdb) dataset from Hugging Face Datasets:

- 25,000 labeled training reviews
- 25,000 labeled test reviews
- Binary labels: `negative` and `positive`

The training script creates a validation split from the original training set and defaults to a practical subset for faster local experimentation.

## Model

The default model is [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased), loaded with `AutoModelForSequenceClassification`.

DistilBERT is a smaller, faster distilled version of BERT. A classification head is added on top of the pretrained encoder and fine-tuned end to end for IMDb sentiment classification.

## Project Structure

```text
.
|-- train.py
|-- evaluate.py
|-- inference.py
|-- requirements.txt
|-- README.md
`-- results/
    `-- .gitkeep
```

Generated artifacts:

- `models/imdb-distilbert/` stores the fine-tuned model and tokenizer.
- `checkpoints/` stores intermediate Trainer checkpoints.
- `results/metrics.json` stores baseline and fine-tuned metrics from training.
- `results/confusion_matrix.png` stores the test-set confusion matrix.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Run a laptop-friendly experiment:

```bash
python train.py \
  --train-samples 12000 \
  --eval-samples 3000 \
  --test-samples 5000 \
  --epochs 2 \
  --batch-size 16
```

Use the full IMDb dataset by passing `-1` for sample counts:

```bash
python train.py --train-samples -1 --eval-samples -1 --test-samples -1 --epochs 3
```

The training script:

1. Loads IMDb from Hugging Face Datasets.
2. Tokenizes reviews with truncation and dynamic padding.
3. Evaluates a baseline pretrained encoder with an untrained sequence-classification head.
4. Fine-tunes DistilBERT with the Hugging Face `Trainer`.
5. Saves the model locally.
6. Writes metrics and a confusion matrix to `results/`.

## Evaluate

Evaluate a saved model:

```bash
python evaluate.py --model-dir models/imdb-distilbert --test-samples 5000
```

## Inference

Classify a raw movie review:

```bash
python inference.py \
  --model-dir models/imdb-distilbert \
  --text "The performances were excellent and the story stayed with me."
```

Example output:

```json
{
  "confidence": 0.98,
  "label": "positive",
  "probabilities": {
    "negative": 0.02,
    "positive": 0.98
  },
  "text": "The performances were excellent and the story stayed with me."
}
```

## Results

After training, `train.py` writes the measured results to `results/metrics.json`. Update this table with your run values before publishing the repository.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Pretrained DistilBERT + untrained classifier head | See `results/metrics.json` | See `results/metrics.json` | See `results/metrics.json` | See `results/metrics.json` |
| Fine-tuned DistilBERT | See `results/metrics.json` | See `results/metrics.json` | See `results/metrics.json` | See `results/metrics.json` |

The baseline uses the same pretrained DistilBERT encoder before task fine-tuning. Because the sequence-classification head has not learned IMDb labels yet, it should perform near chance; fine-tuning should provide the major gain.

## Confusion Matrix

After running training or evaluation, the confusion matrix image is saved to `results/confusion_matrix.png`.

> **Note:** The confusion matrix is generated at runtime and is not tracked in this repository. Run `train.py` or `evaluate.py` to produce it locally.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Evaluate
- scikit-learn
- Matplotlib and Seaborn
