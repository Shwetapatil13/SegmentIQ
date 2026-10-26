"""
Data loading and header normalization module for customer segmentation.
"""

import os
import pandas as pd
from typing import Union, BinaryIO, Tuple, Dict, Any


HEADER_MAPPINGS = {
    "genre": "Gender",
    "gender": "Gender",
    "customerid": "CustomerID",
    "customer_id": "CustomerID",
    "id": "CustomerID",
    "age": "Age",
    "annual income (k$)": "Annual Income (k$)",
    "annual_income_(k$)": "Annual Income (k$)",
    "annual_income": "Annual Income (k$)",
    "annual income": "Annual Income (k$)",
    "spending score (1-100)": "Spending Score (1-100)",
    "spending_score_(1-100)": "Spending Score (1-100)",
    "spending score": "Spending Score (1-100)",
    "spending_score": "Spending Score (1-100)"
}


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names across common variations (e.g. 'Genre' -> 'Gender').
    Preserves original column order while normalizing aliases.
    """
    renamed_cols = {}
    for col in df.columns:
        cleaned = str(col).strip().lower()
        if cleaned in HEADER_MAPPINGS:
            renamed_cols[col] = HEADER_MAPPINGS[cleaned]
    return df.rename(columns=renamed_cols)


def load_dataset(source: Union[str, BinaryIO] = "data/Mall_Customers.csv") -> Tuple[pd.DataFrame, bool]:
    """
    Load dataset from path or file buffer, validate, and normalize headers.
    Returns:
        (df, is_synthetic_flag)
    """
    is_synthetic = False
    
    if isinstance(source, str):
        # Resolve path across multiple execution roots (root, notebooks/, tests/)
        if not os.path.exists(source):
            base_filename = os.path.basename(source)
            candidate_paths = [
                os.path.join("data", base_filename),
                os.path.join("..", "data", base_filename),
                os.path.join(".", base_filename)
            ]
            
            # If the user requested default Mall_Customers, also check root or synthetic
            if "mall_customers" in base_filename.lower():
                candidate_paths.extend([
                    "Mall_Customers.csv",
                    os.path.join("data", "Mall_Customers.csv"),
                    os.path.join("..", "data", "Mall_Customers.csv"),
                    os.path.join("data", "sample_synthetic.csv"),
                    os.path.join("..", "data", "sample_synthetic.csv")
                ])
                
            found = False
            for cand in candidate_paths:
                if os.path.exists(cand):
                    source = cand
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Requested dataset file '{source}' does not exist.")
                
        if "sample_synthetic" in str(source):
            is_synthetic = True
            
        try:
            df = pd.read_csv(source)
        except Exception as e:
            raise ValueError(f"Failed to read CSV from '{source}': {str(e)}")
    else:
        try:
            df = pd.read_csv(source)
        except Exception as e:
            raise ValueError(f"Failed to parse uploaded CSV: {str(e)}")

    if df.empty:
        raise ValueError("The provided dataset is empty (0 rows).")

    df = normalize_headers(df)
    return df, is_synthetic


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect dataset shape, data types, missing value statistics, and duplicate rows.
    """
    n_rows, n_cols = df.shape
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / n_rows) * 100
    n_duplicates = int(df.duplicated().sum())

    missing_df = pd.DataFrame({
        "Missing Values": missing_counts,
        "Missing %": missing_pcts.round(2)
    })

    dtypes_df = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Non-Null Count": df.notnull().sum(),
        "Unique Values": df.nunique()
    })

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_duplicates": n_duplicates,
        "missing_summary": missing_df,
        "dtypes_summary": dtypes_df,
        "descriptive_stats": df.describe(include="all").T
    }
