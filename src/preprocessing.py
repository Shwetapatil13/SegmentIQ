"""
Preprocessing, encoding, missing-value imputation, and feature standardization.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


DEFAULT_ID_COLUMNS = ["CustomerID", "customer_id", "id", "CUST_ID"]


def validate_feature_selection(df: pd.DataFrame, selected_features: List[str]) -> None:
    """
    Validate that selected features are viable for clustering.
    Rejects fewer than 2 features, too few rows, or constant features.
    """
    if len(selected_features) < 2:
        raise ValueError(
            f"Clustering requires at least 2 features, but only {len(selected_features)} was selected: {selected_features}."
        )

    if len(df) < 5:
        raise ValueError(
            f"Dataset has only {len(df)} rows. Clustering requires at least 5 observations."
        )

    for col in selected_features:
        if col not in df.columns:
            raise ValueError(f"Selected feature '{col}' is not present in the dataset.")
        
        # Check if constant
        if df[col].nunique(dropna=False) <= 1:
            raise ValueError(
                f"Feature '{col}' is constant (zero variance). Constant columns provide no discriminative information and must be removed."
            )


class CustomerPreprocessor:
    """
    Reusable preprocessor that standardizes numeric features, optionally encodes categorical features,
    and imputes missing values. Fitted once and reused for centroids and inference.
    """

    def __init__(
        self,
        selected_features: List[str],
        missing_strategy: str = "impute",  # 'impute' or 'drop'
        id_columns: Optional[List[str]] = None
    ):
        self.selected_features = list(selected_features)
        self.missing_strategy = missing_strategy
        self.id_columns = id_columns or DEFAULT_ID_COLUMNS
        
        # Internal state
        self.numeric_features_: List[str] = []
        self.categorical_features_: List[str] = []
        self.transformed_feature_names_: List[str] = []
        self.fitted_scaler_: Optional[StandardScaler] = None
        self.fitted_encoder_: Optional[OneHotEncoder] = None
        self.imputers_: Dict[str, Any] = {}
        self.is_fitted_: bool = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
        """
        Fit preprocessing transformers and return scaled matrix, cleaned DataFrame, and audit metadata.
        """
        validate_feature_selection(df, self.selected_features)

        audit_log = {
            "initial_rows": len(df),
            "rows_dropped": 0,
            "missing_values_imputed": 0,
            "selected_features": self.selected_features,
            "id_columns_excluded": [col for col in self.id_columns if col in df.columns]
        }

        # Work on selected features
        working_df = df[self.selected_features].copy()

        # Handle missing values
        if self.missing_strategy == "drop":
            rows_before = len(working_df)
            working_df = working_df.dropna()
            audit_log["rows_dropped"] = rows_before - len(working_df)
            if len(working_df) < 5:
                raise ValueError("Dropping missing values left fewer than 5 rows. Use 'impute' strategy instead.")
        else:
            # Impute strategy: median for numeric, most_frequent for categorical
            for col in self.selected_features:
                null_count = int(working_df[col].isnull().sum())
                if null_count > 0:
                    audit_log["missing_values_imputed"] += null_count
                    if pd.api.types.is_numeric_dtype(working_df[col]):
                        imp = SimpleImputer(strategy="median")
                        working_df[col] = imp.fit_transform(working_df[[col]]).ravel()
                        self.imputers_[col] = imp
                    else:
                        imp = SimpleImputer(strategy="most_frequent")
                        working_df[col] = imp.fit_transform(working_df[[col]]).ravel()
                        self.imputers_[col] = imp

        # Identify numeric vs categorical
        self.numeric_features_ = [
            c for c in self.selected_features if pd.api.types.is_numeric_dtype(working_df[c])
        ]
        self.categorical_features_ = [
            c for c in self.selected_features if c not in self.numeric_features_
        ]

        # Fit Scaler on numeric features
        if self.numeric_features_:
            self.fitted_scaler_ = StandardScaler()
            scaled_num = self.fitted_scaler_.fit_transform(working_df[self.numeric_features_])
        else:
            scaled_num = np.empty((len(working_df), 0))

        # Fit Encoder on categorical features if any
        if self.categorical_features_:
            self.fitted_encoder_ = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            encoded_cat = self.fitted_encoder_.fit_transform(working_df[self.categorical_features_])
            cat_names = self.fitted_encoder_.get_feature_names_out(self.categorical_features_).tolist()
        else:
            encoded_cat = np.empty((len(working_df), 0))
            cat_names = []

        self.transformed_feature_names_ = self.numeric_features_ + cat_names
        X_scaled = np.hstack([scaled_num, encoded_cat]) if encoded_cat.shape[1] > 0 else scaled_num

        self.is_fitted_ = True
        return X_scaled, working_df, audit_log

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new observations using the ALREADY FITTED scaler and encoder (no refitting!).
        """
        if not self.is_fitted_:
            raise RuntimeError("CustomerPreprocessor must be fitted before calling transform().")

        working_df = df[self.selected_features].copy()

        # Apply fitted imputers
        for col, imp in self.imputers_.items():
            if col in working_df.columns:
                working_df[col] = imp.transform(working_df[[col]]).ravel()

        if self.numeric_features_:
            scaled_num = self.fitted_scaler_.transform(working_df[self.numeric_features_])
        else:
            scaled_num = np.empty((len(working_df), 0))

        if self.categorical_features_:
            encoded_cat = self.fitted_encoder_.transform(working_df[self.categorical_features_])
        else:
            encoded_cat = np.empty((len(working_df), 0))

        return np.hstack([scaled_num, encoded_cat]) if encoded_cat.shape[1] > 0 else scaled_num

    def inverse_transform_numeric(self, scaled_array: np.ndarray) -> np.ndarray:
        """
        Inverse-transform numeric features back to original measurement units (e.g. for K-Means centroids).
        """
        if not self.is_fitted_ or self.fitted_scaler_ is None:
            raise RuntimeError("Scaler is not fitted.")
        # Only numeric slice
        n_num = len(self.numeric_features_)
        return self.fitted_scaler_.inverse_transform(scaled_array[:, :n_num])


def compute_before_after_summary(raw_df: pd.DataFrame, X_scaled: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Generate side-by-side comparison of raw statistics vs scaled statistics.
    """
    records = []
    for i, col in enumerate(feature_names):
        if col in raw_df.columns and pd.api.types.is_numeric_dtype(raw_df[col]):
            raw_mean = float(raw_df[col].mean())
            raw_std = float(raw_df[col].std())
            raw_min = float(raw_df[col].min())
            raw_max = float(raw_df[col].max())
        else:
            raw_mean, raw_std, raw_min, raw_max = np.nan, np.nan, np.nan, np.nan

        scaled_mean = float(np.mean(X_scaled[:, i]))
        scaled_std = float(np.std(X_scaled[:, i]))
        scaled_min = float(np.min(X_scaled[:, i]))
        scaled_max = float(np.max(X_scaled[:, i]))

        records.append({
            "Feature": col,
            "Raw Mean": round(raw_mean, 2) if not np.isnan(raw_mean) else "N/A",
            "Raw Std": round(raw_std, 2) if not np.isnan(raw_std) else "N/A",
            "Raw Min": round(raw_min, 2) if not np.isnan(raw_min) else "N/A",
            "Raw Max": round(raw_max, 2) if not np.isnan(raw_max) else "N/A",
            "Scaled Mean": round(scaled_mean, 4),
            "Scaled Std": round(scaled_std, 4),
            "Scaled Min": round(scaled_min, 2),
            "Scaled Max": round(scaled_max, 2)
        })
    return pd.DataFrame(records)
