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
    r"\b(km|meter|meters|miles?|near|around|north|south|east|west|from)\b",
    re.IGNORECASE,
)


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
        default="Babelscape/wikineural-multilingual-ner",
        help="HF token-classification model.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def normalize_entity_text(text: str) -> str:
    # Normalize spacing and strip punctuation fragments from entity boundaries.
    text = re.sub(r"\s+", " ", text).strip(" ,.;:!?()[]{}\"'")
    return text


def keep_entity(candidate: str) -> bool:
    if not candidate:
        return False
    # Drop address-like candidates and unit-bearing fragments.
    if any(ch.isdigit() for ch in candidate):
        return False
    if RELATIVE_HINT_RE.search(candidate):
        return False
    return True


def build_lexicon(train_file: Path) -> List[str]:
    if not train_file.exists():
        return []

    df = pd.read_csv(train_file)
    if "location" not in df.columns:
        return []

    # Treat each full label string as a candidate place phrase.
    phrases = (
        df["location"]
        .fillna("")
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )

    # Longest-first matching avoids splitting multi-token place names.
    phrases.sort(key=len, reverse=True)
    return phrases


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
            end = idx + len(phrase)
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
    deduped = list(dict.fromkeys(ordered_entities))
    return deduped


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
                label = ent.get("entity_group", "")
                if label.upper() not in {"LOC", "LOCATION"}:
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
