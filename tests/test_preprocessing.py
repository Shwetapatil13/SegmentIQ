"""
Unit tests for preprocessing, encoding, scaling, and validations.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import (
    CustomerPreprocessor,
    validate_feature_selection,
    compute_before_after_summary
)


def test_validate_feature_selection_rejects_single_feature():
    df = pd.DataFrame({"Age": [20, 30, 40, 50, 60], "Income": [10, 20, 30, 40, 50]})
    with pytest.raises(ValueError, match="at least 2 features"):
        validate_feature_selection(df, ["Age"])


def test_validate_feature_selection_rejects_too_few_rows():
    df = pd.DataFrame({"Age": [20, 30], "Income": [10, 20]})
    with pytest.raises(ValueError, match="at least 5 observations"):
        validate_feature_selection(df, ["Age", "Income"])


def test_validate_feature_selection_rejects_constant_column():
    df = pd.DataFrame({
        "Age": [25, 25, 25, 25, 25],
        "Income": [10, 20, 30, 40, 50]
    })
    with pytest.raises(ValueError, match="is constant"):
        validate_feature_selection(df, ["Age", "Income"])


def test_scaling_mean_zero_std_one():
    df = pd.DataFrame({
        "Age": [19, 25, 35, 45, 55, 65],
        "Annual Income (k$)": [15, 30, 50, 75, 90, 120]
    })
    preprocessor = CustomerPreprocessor(selected_features=["Age", "Annual Income (k$)"])
    X_scaled, working_df, audit = preprocessor.fit_transform(df)

    # Scaled features must have mean approx 0 and std approx 1
    np.testing.assert_allclose(np.mean(X_scaled, axis=0), [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(np.std(X_scaled, axis=0), [1.0, 1.0], atol=1e-7)


def test_missing_value_impute_strategy():
    df = pd.DataFrame({
        "Age": [20.0, 30.0, np.nan, 40.0, 50.0],
        "Income": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    preprocessor = CustomerPreprocessor(selected_features=["Age", "Income"], missing_strategy="impute")
    X_scaled, working_df, audit = preprocessor.fit_transform(df)

    assert audit["missing_values_imputed"] == 1
    # Check that imputed value in working_df equals median of [20, 30, 40, 50] which is 35.0
    assert working_df.loc[2, "Age"] == 35.0


def test_missing_value_drop_strategy():
    df = pd.DataFrame({
        "Age": [20.0, 30.0, np.nan, 40.0, 50.0, 60.0],
        "Income": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    })
    preprocessor = CustomerPreprocessor(selected_features=["Age", "Income"], missing_strategy="drop")
    X_scaled, working_df, audit = preprocessor.fit_transform(df)

    assert audit["rows_dropped"] == 1
    assert len(working_df) == 5


def test_categorical_encoding():
    df = pd.DataFrame({
        "Gender": ["Male", "Female", "Female", "Male", "Female"],
        "Income": [20, 30, 40, 50, 60]
    })
    preprocessor = CustomerPreprocessor(selected_features=["Gender", "Income"])
    X_scaled, working_df, audit = preprocessor.fit_transform(df)

    # Income is numeric, Gender is encoded as 1 column (drop='first')
    assert X_scaled.shape == (5, 2)


def test_preprocessor_reused_without_refitting():
    df_train = pd.DataFrame({
        "Age": [20, 30, 40, 50, 60],
        "Income": [10, 20, 30, 40, 50]
    })
    preprocessor = CustomerPreprocessor(selected_features=["Age", "Income"])
    X_scaled, _, _ = preprocessor.fit_transform(df_train)

    df_test = pd.DataFrame({
        "Age": [30, 50],
        "Income": [20, 40]
    })
    X_test_scaled = preprocessor.transform(df_test)
    assert X_test_scaled.shape == (2, 2)

    # Check inverse transform on centroids
    recovered = preprocessor.inverse_transform_numeric(X_scaled)
    np.testing.assert_allclose(recovered, df_train.values, atol=1e-5)
