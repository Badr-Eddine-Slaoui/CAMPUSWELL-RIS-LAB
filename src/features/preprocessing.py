import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    OrdinalEncoder,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from src.config import (
    NUMERICAL_FEATURES,
    ENGINEERED_NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    NOMINAL_FEATURES,
    BINARY_FEATURES,
    PHYSICAL_ACTIVITY_ORDER,
    DIET_QUALITY_ORDER,
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            X_out = X.copy()
        else:
            X_out = pd.DataFrame(X)
        if "Sleep_Hours" in X_out.columns and "Screen_Time_Hours" in X_out.columns:
            X_out["sleep_to_screen_ratio"] = X_out["Sleep_Hours"] / (X_out["Screen_Time_Hours"] + 0.1)
        return X_out


class ColumnWiseImputer(BaseEstimator, TransformerMixin):
    def __init__(self, median_cols=None, iterative_cols=None, random_state=42):
        self.median_cols = median_cols or []
        self.iterative_cols = iterative_cols or []
        self.random_state = random_state
        self.median_imputer = None
        self.iterative_imputer = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            X_df = X
        else:
            X_df = pd.DataFrame(X)
            self.feature_names_in_ = list(range(X.shape[1]))

        if self.median_cols:
            actual_median = [c for c in self.median_cols if c in X_df.columns]
            if actual_median:
                self.median_imputer = SimpleImputer(strategy="median")
                self.median_imputer.fit(X_df[actual_median])

        if self.iterative_cols:
            actual_it = [c for c in self.iterative_cols if c in X_df.columns]
            if actual_it:
                self.iterative_imputer = IterativeImputer(random_state=self.random_state, max_iter=10)
                self.iterative_imputer.fit(X_df[actual_it])

        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        if self.median_imputer and self.median_cols:
            actual_median = [c for c in self.median_cols if c in X_df.columns]
            if actual_median:
                X_df[actual_median] = self.median_imputer.transform(X_df[actual_median])
