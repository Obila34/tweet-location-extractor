import argparse
import json
from pathlib import Path

import pandas as pd

from utils import clean_tweet_text, infer_columns, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess train/test tweet data.")
    parser.add_argument("--data-dir", type=str, default="DATA", help="Input data folder.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/processed",
        help="Folder for cleaned train/test CSV files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _choose_file(data_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files were found in {data_dir}: {candidates}")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support common naming variants to reduce manual edits.
    train_path = _choose_file(data_dir, ["Train_1.csv", "Train.csv", "train.csv"])
    test_path = _choose_file(data_dir, ["Test.csv", "test.csv"])

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_cols = infer_columns(train_df, with_location=True)
    test_cols = infer_columns(test_df, with_location=False)

    # Build canonical train frame: tweet_id, text, location.
    clean_train = pd.DataFrame(
        {
            "tweet_id": train_df[train_cols["id"]].astype(str),
            "text": train_df[train_cols["text"]].fillna("").map(clean_tweet_text),
            "location": train_df[train_cols["location"]].fillna("").astype(str).str.strip(),
        }
    )

    # Build canonical test frame: tweet_id, text.
    clean_test = pd.DataFrame(
        {
            "tweet_id": test_df[test_cols["id"]].astype(str),
            "text": test_df[test_cols["text"]].fillna("").map(clean_tweet_text),
        }
    )

    # Drop exact duplicates to stabilize training/inference behavior.
    clean_train = clean_train.drop_duplicates(subset=["tweet_id", "text", "location"]).reset_index(drop=True)
    clean_test = clean_test.drop_duplicates(subset=["tweet_id", "text"]).reset_index(drop=True)

    train_out = output_dir / "train_clean.csv"
    test_out = output_dir / "test_clean.csv"
    clean_train.to_csv(train_out, index=False)
    clean_test.to_csv(test_out, index=False)

    meta = {
        "seed": args.seed,
        "input": {"train": str(train_path), "test": str(test_path)},
        "rows": {"train": len(clean_train), "test": len(clean_test)},
        "columns": {
            "train_inferred": train_cols,
            "test_inferred": test_cols,
            "train_output": ["tweet_id", "text", "location"],
            "test_output": ["tweet_id", "text"],
        },
    }
    with open(output_dir / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {train_out}")
    print(f"Saved: {test_out}")


if __name__ == "__main__":
    main()
