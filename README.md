# disaster-tweet-ner

Location Mention Recognition (LMR) baseline pipeline for disaster tweets.

## Recommended approach

For this challenge, the best accuracy vs simplicity tradeoff is a **hybrid transformer NER pipeline**:

1. Use a pretrained token-classification transformer (`Babelscape/wikineural-multilingual-ner`) to detect location spans.
2. Add a lightweight lexicon matcher mined from training labels to recover missed locations.
3. Apply deterministic post-processing rules to remove address/relative fragments and keep place names in order.

Why this approach:
- Strong baseline quality without lengthy training.
- Works within compute limits (T4, <8h training budget not required for this baseline).
- Simple to maintain, reproducible, and easy to iterate.

If you want maximum leaderboard performance later, you can replace this with a fine-tuned token-classification model trained on BIO tags; the rest of this project structure will still work.

## Folder structure

```text
disaster-tweet-ner/
├── DATA/
│   ├── Train_1.csv
│   ├── Test.csv
│   └── SampleSubmission.csv
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── preprocess.py
│   ├── predict.py
│   └── generate_submission.py
├── outputs/
│   ├── processed/
│   ├── predictions/
│   └── submissions/
├── models/
├── logs/
├── requirements.txt
└── README.md
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Hardware and runtime

- Recommended: NVIDIA T4 GPU (16 GB VRAM), 4+ CPU cores, 16+ GB RAM.
- Inference runtime estimate (test-size dependent): ~5-35 minutes on T4.
- Preprocessing and submission generation: usually <2 minutes.
- This baseline performs no model fine-tuning, so it is comfortably within the challenge time constraints.

## Data assumptions

Expected canonical columns after preprocessing:
- `tweet_id`
- `text`
- `location` (train only)

The scripts automatically infer common column variants (for example `id` vs `tweet_id`, `location` vs `locations`).

## Run full pipeline

Run from project root.

1. Preprocess and clean data:

```bash
python src/preprocess.py --data-dir DATA --output-dir outputs/processed --seed 42
```

2. Predict location mentions on cleaned test set:

```bash
python src/predict.py \
	--input-file outputs/processed/test_clean.csv \
	--train-file outputs/processed/train_clean.csv \
	--output-file outputs/predictions/test_predictions.csv \
	--model-name Babelscape/wikineural-multilingual-ner \
	--batch-size 32 \
	--seed 42
```

3. Generate challenge submission CSV:

```bash
python src/generate_submission.py \
	--pred-file outputs/predictions/test_predictions.csv \
	--sample-submission DATA/SampleSubmission.csv \
	--output-file outputs/submissions/submission.csv
```

Final file for upload:
- `outputs/submissions/submission.csv`

## Notes on output format

- Predictions are generated as space-separated location mentions in appearance order.
- Tweets with no extracted locations get an empty string.
- Simple filters are applied to avoid address/relative patterns (for example number-based fragments).

## Reproducibility

All scripts expose a `--seed` argument and set global random seeds for Python/NumPy/PyTorch.