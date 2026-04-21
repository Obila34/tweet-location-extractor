import os
import random
import re
import unicodedata
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def set_global_seed(seed: int = 42) -> None:
    """Set all supported random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Import torch lazily so preprocessing can run even if torch is unavailable.
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


ID_CANDIDATES = ["tweet_id", "id", "ID"]
TEXT_CANDIDATES = ["text", "tweet", "message"]
LOCATION_CANDIDATES = ["location", "locations", "label", "target"]


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in columns}
    for name in candidates:
        match = lower_to_original.get(name.lower())
        if match is not None:
            return match
    return None


def infer_columns(df: pd.DataFrame, with_location: bool) -> Dict[str, str]:
    """Infer canonical column names from a dataframe."""
    id_col = _find_column(df.columns, ID_CANDIDATES)
    text_col = _find_column(df.columns, TEXT_CANDIDATES)

    if id_col is None or text_col is None:
        raise ValueError(
            f"Could not infer required columns from {list(df.columns)}. "
            "Expected an ID-like and text-like column."
        )

    mapping = {"id": id_col, "text": text_col}

    if with_location:
        loc_col = _find_column(df.columns, LOCATION_CANDIDATES)
        if loc_col is None:
            raise ValueError(
                f"Could not infer location column from {list(df.columns)}."
            )
        mapping["location"] = loc_col

    return mapping


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_tweet_text(text: str) -> str:
    """Normalize noisy tweet text while preserving location-bearing tokens."""
    if not isinstance(text, str):
        text = ""

    # Unicode normalization helps with mixed-width and compatibility characters.
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs and user mentions; they are generally not useful for place extraction.
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)

    # Keep hashtag words but remove the hash marker itself.
    text = text.replace("#", "")

    # Collapse repeating whitespace and trim boundaries.
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text
