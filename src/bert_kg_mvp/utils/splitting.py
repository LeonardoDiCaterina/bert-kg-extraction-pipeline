import re
from typing import Tuple
import pandas as pd


def extract_company_identifier(doc_id: str) -> str:
    """
    Extracts a clean company identifier or ticker from a document ID or filename.
    Examples:
        'AAPL_2025.pdf' -> 'AAPL'
        'sec-edgar-filings/MSFT/10-K/...' -> 'MSFT'
        'NVDA' -> 'NVDA'
    """
    clean = str(doc_id).replace("\\", "/")

    # Check for sec-edgar-filings/<TICKER>/...
    if "sec-edgar-filings/" in clean:
        parts = clean.split("sec-edgar-filings/")[1].split("/")
        if parts and parts[0]:
            return parts[0].upper()

    # Check for basename like AAPL_2025 or AAPL-10K
    filename = clean.split("/")[-1]
    match = re.match(r"^([A-Za-z0-9]+)[\_\-\.]", filename)
    if match:
        return match.group(1).upper()

    return filename.upper()


def split_by_company(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    company_col: str = "doc_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions a DataFrame into train, validation, and test splits by grouping entire
    companies together. This ensures that validation and test sets contain completely
    unseen companies (out-of-distribution company evaluation).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    # Normalize split ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / total
    val_ratio = val_ratio / total

    # Extract company identifier for each row
    df_copy = df.copy()
    if "company_id" not in df_copy.columns:
        df_copy["company_id"] = df_copy[company_col].apply(extract_company_identifier)

    unique_companies = sorted(df_copy["company_id"].unique())
    n_companies = len(unique_companies)

    if n_companies < 3:
        # Edge case: If fewer than 3 companies, fallback to row-level random split
        shuffled = df_copy.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_df = shuffled.iloc[:n_train]
        val_df = shuffled.iloc[n_train:n_train + n_val]
        test_df = shuffled.iloc[n_train + n_val:]
        return train_df, val_df, test_df

    # Shuffle company identifiers
    companies_series = pd.Series(unique_companies).sample(frac=1, random_state=seed).tolist()

    n_train_comp = max(1, int(round(n_companies * train_ratio)))
    n_val_comp = max(1, int(round(n_companies * val_ratio)))
    # Ensure at least 1 company in test if possible
    if n_train_comp + n_val_comp >= n_companies:
        n_train_comp = max(1, n_companies - 2)
        n_val_comp = 1

    train_comps = set(companies_series[:n_train_comp])
    val_comps = set(companies_series[n_train_comp:n_train_comp + n_val_comp])
    test_comps = set(companies_series[n_train_comp + n_val_comp:])

    train_df = df_copy[df_copy["company_id"].isin(train_comps)].copy()
    val_df = df_copy[df_copy["company_id"].isin(val_comps)].copy()
    test_df = df_copy[df_copy["company_id"].isin(test_comps)].copy()

    return train_df, val_df, test_df
