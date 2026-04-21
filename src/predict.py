import argparse
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from utils import set_global_seed


RELATIVE_HINT_RE = re.compile(
    r"\b(km|meter|meters|miles?|near|around|north|south|east|west|from|away|towards?)\b",
    re.IGNORECASE,
)

# Short words that are frequently hallucinated as locations by generic NER models.
NOISE_WORDS = {
    "in",
    "or",
    "and",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "on",
    "at",
    "by",
    "with",
    "from",
    "near",
    "new",
    "red",
    "wildfire",
    "flood",
    "storm",
    "disaster",
    "pic",
    "ni",
    "ne",
    "pr",
    "sa",
    "cal",
}

# Keep only a tiny set of short uppercase geo abbreviations to improve precision.
ALLOWED_SHORT = {"US", "USA", "UK", "UAE", "EU", "NYC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict location mentions from tweets.")
    parser.add_argument(
        "--input-file",
        type=str,
        default="outputs/processed/test_clean.csv",
        help="Path to canonical test CSV with columns tweet_id,text.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="outputs/processed/train_clean.csv",
        help="Optional cleaned train CSV for lexicon augmentation.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="outputs/predictions/test_predictions.csv",
        help="Where to write predictions.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dslim/bert-base-NER",
        help="HF token-classification model.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--min-score", type=float, default=0.80, help="Minimum entity confidence.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def normalize_entity_text(text: str) -> str:
    # Normalize spacing and strip punctuation fragments from entity boundaries.
    return re.sub(r"\s+", " ", text).strip(" ,.;:!?()[]{}\"'`“”’")


def _is_title_or_upper(word: str) -> bool:
    return bool(re.match(r"^[A-Z][a-z]+$", word) or re.match(r"^[A-Z]{2,}$", word))


def keep_entity(candidate: str) -> bool:
    candidate = normalize_entity_text(candidate)
    if not candidate:
        return False

    if any(ch.isdigit() for ch in candidate):
        return False
    if RELATIVE_HINT_RE.search(candidate):
        return False

    lc = candidate.lower()
    if lc in NOISE_WORDS:
        return False

    words = [w for w in re.split(r"\s+", candidate) if w]
    if not words:
        return False

    # Reject tiny tokens unless they are in a small explicit whitelist.
    if len(candidate) <= 3 and candidate.upper() not in ALLOWED_SHORT:
        return False

    # Reject all-lowercase single words; true places are usually proper nouns.
    if len(words) == 1 and words[0].islower():
        return False

    # Require at least one token to look like a proper noun or uppercase abbreviation.
    if not any(_is_title_or_upper(w) for w in words):
        return False

    return True


def build_lexicon(train_file: Path) -> List[str]:
    if not train_file.exists():
        return []

    df = pd.read_csv(train_file)
    if "location" not in df.columns:
        return []

    phrases = (
        df["location"]
        .fillna("")
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )

    # Filter low-quality phrases from training labels to prevent noisy matching.
    filtered = [p for p in phrases if keep_entity(p)]
    filtered.sort(key=len, reverse=True)
    return filtered


def find_lexicon_matches(text: str, lexicon: List[str]) -> List[Tuple[int, int, str]]:
    matches: List[Tuple[int, int, str]] = []
    lower = text.lower()
    for phrase in lexicon:
        p = phrase.lower()
        start = 0
        while True:
            idx = lower.find(p, start)
            if idx == -1:
                break

            # Simple token-boundary guard to reduce substring false matches.
            left_ok = idx == 0 or not text[idx - 1].isalnum()
            end = idx + len(phrase)
            right_ok = end == len(text) or not text[end].isalnum()
            if left_ok and right_ok:
                matches.append((idx, end, text[idx:end]))

            start = end
    return matches


def merge_spans(spans: List[Tuple[int, int, str]]) -> List[str]:
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int, str]] = []
    for s in spans:
        if not merged or s[0] >= merged[-1][1]:
            merged.append(s)
        else:
            # Keep longer span when overlaps occur.
            prev = merged[-1]
            if (s[1] - s[0]) > (prev[1] - prev[0]):
                merged[-1] = s

    ordered_entities = []
    for _, _, txt in merged:
        txt = normalize_entity_text(txt)
        if keep_entity(txt):
            ordered_entities.append(txt)

    # Remove duplicates while preserving first appearance order.
    return list(dict.fromkeys(ordered_entities))


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    input_path = Path(args.input_file)
    train_path = Path(args.train_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "tweet_id" not in df.columns or "text" not in df.columns:
        raise ValueError(f"Expected columns tweet_id,text in {input_path}")

    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name)

    ner = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )

    lexicon = build_lexicon(train_path)

    predictions = []
    texts = df["text"].fillna("").astype(str).tolist()
    ids = df["tweet_id"].astype(str).tolist()

    # Batch through tweets for better throughput on GPU.
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Predicting"):
        batch_texts = texts[i : i + args.batch_size]
        batch_ids = ids[i : i + args.batch_size]

        ner_outputs = ner(batch_texts)
        for tweet_id, text, entities in zip(batch_ids, batch_texts, ner_outputs):
            spans: List[Tuple[int, int, str]] = []

            for ent in entities:
                label = str(ent.get("entity_group", "")).upper()
                if label not in {"LOC", "LOCATION", "GPE"}:
                    continue

                score = float(ent.get("score", 0.0))
                if score < args.min_score:
                    continue

                start = int(ent.get("start", -1))
                end = int(ent.get("end", -1))
                word = str(ent.get("word", ""))

                if start >= 0 and end > start:
                    spans.append((start, end, text[start:end]))
                elif word:
                    # Fallback path for models that omit char offsets.
                    idx = text.lower().find(word.lower())
                    if idx >= 0:
                        spans.append((idx, idx + len(word), text[idx : idx + len(word)]))

            # Add deterministic lexicon matches mined from training labels.
            spans.extend(find_lexicon_matches(text, lexicon))

            locations = merge_spans(spans)
            predictions.append({"tweet_id": tweet_id, "location": " ".join(locations)})

    out_df = pd.DataFrame(predictions)
    out_df.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    main()
