
from typing import List, Tuple
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = [c for c in self.columns if c in X.columns]
        return X[cols].copy()

def split_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col] if target_col in df.columns else None
    X = df.drop(columns=[target_col], errors='ignore')
    return X, y
