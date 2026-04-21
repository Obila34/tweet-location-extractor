import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate challenge submission CSV.")
    parser.add_argument(
        "--pred-file",
        type=str,
        default="outputs/predictions/test_predictions.csv",
        help="Predictions file with tweet_id and location columns.",
    )
    parser.add_argument(
        "--sample-submission",
        type=str,
        default="DATA/SampleSubmission.csv",
        help="Sample submission used to enforce expected column names/order.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="outputs/submissions/submission.csv",
        help="Final submission output path.",
    )
    return parser.parse_args()


def choose_columns(sample_df: pd.DataFrame) -> tuple[str, str]:
    cols = list(sample_df.columns)
    lower = {c.lower(): c for c in cols}

    id_col = lower.get("tweet_id") or lower.get("id")
    loc_col = lower.get("location") or lower.get("locations")

    if id_col is None or loc_col is None:
        raise ValueError(
            f"Could not infer id/location columns from sample submission columns: {cols}"
        )

    return id_col, loc_col


def main() -> None:
    args = parse_args()

    pred_path = Path(args.pred_file)
    sample_path = Path(args.sample_submission)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(pred_path)
    sample_df = pd.read_csv(sample_path)

    if "tweet_id" not in pred_df.columns or "location" not in pred_df.columns:
        raise ValueError("Prediction file must contain tweet_id and location columns.")

    id_col, loc_col = choose_columns(sample_df)

    # Join against sample IDs so row order matches expected submission order.
    sub = sample_df[[id_col]].merge(
        pred_df[["tweet_id", "location"]],
        left_on=id_col,
        right_on="tweet_id",
        how="left",
    )

    sub[loc_col] = sub["location"].fillna("").astype(str)
    sub = sub[[id_col, loc_col]]

    sub.to_csv(output_path, index=False)
    print(f"Saved submission: {output_path}")


if __name__ == "__main__":
    main()
