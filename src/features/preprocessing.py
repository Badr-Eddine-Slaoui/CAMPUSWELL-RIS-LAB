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
