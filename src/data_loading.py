
import pandas as pd

def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def coerce_datetime(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df
