"""
Unit tests for data loader and header normalization.
"""

import pytest
import pandas as pd
from io import StringIO
from src.data_loader import normalize_headers, load_dataset, get_dataset_summary


def test_normalize_headers():
    df = pd.DataFrame({
        "CustomerID": [1, 2],
        "Genre": ["Male", "Female"],
        "Age": [19, 21],
        "Annual Income (k$)": [15, 15],
        "Spending Score (1-100)": [39, 81]
    })
    norm = normalize_headers(df)
    assert "Gender" in norm.columns
    assert "Genre" not in norm.columns


def test_normalize_headers_variations():
    df = pd.DataFrame({
        "customer_id": [1],
        "gender": ["Female"],
        "annual_income": [50],
        "spending_score": [60]
    })
    norm = normalize_headers(df)
    assert "CustomerID" in norm.columns
    assert "Gender" in norm.columns
    assert "Annual Income (k$)" in norm.columns
    assert "Spending Score (1-100)" in norm.columns


def test_load_dataset_from_buffer():
    csv_data = "CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)\n1,Male,20,40,50\n2,Female,22,42,55"
    buf = StringIO(csv_data)
    df, is_synthetic = load_dataset(buf)
    assert len(df) == 2
    assert not is_synthetic
    assert list(df.columns) == ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]


def test_load_dataset_empty_error():
    buf = StringIO("")
    with pytest.raises(ValueError):
        load_dataset(buf)


def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_folder/non_existent_file.csv")


def test_get_dataset_summary():
    df = pd.DataFrame({
        "Age": [20, 30, None],
        "Income": [50, 60, 70]
    })
    summary = get_dataset_summary(df)
    assert summary["n_rows"] == 3
    assert summary["n_cols"] == 2
    assert summary["missing_summary"].loc["Age", "Missing Values"] == 1
    assert summary["missing_summary"].loc["Income", "Missing Values"] == 0
