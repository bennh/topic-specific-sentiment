# src/preprocessing.py

import re
import pandas as pd


REQUIRED_COLUMNS = [
    "Headlines",
    "Time",
    "Description"
]


def load_reuters_data(filepath: str) -> pd.DataFrame:
    """
    Load Reuters financial news dataset.
    """

    df = pd.read_csv(filepath)

    missing_cols = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}"
        )

    return df


def clean_text(text: str) -> str:
    """
    Clean raw news text.
    """

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine headline and description.
    """

    df = df.copy()

    df["text"] = (
        df["Headlines"].fillna("")
        + " "
        + df["Description"].fillna("")
    )

    return df


def convert_time_column(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert Reuters date column.
    """

    df = df.copy()

    df["Time"] = pd.to_datetime(
        df["Time"],
        errors="coerce"
    )

    return df


def preprocess_news(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    """

    df = build_text_column(df)

    df["clean_text"] = (
        df["text"]
        .astype(str)
        .apply(clean_text)
    )

    df = convert_time_column(df)

    return df