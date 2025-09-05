
import pandas as pd

def join_text_columns(df: pd.DataFrame, text_cols):
    cols = [c for c in text_cols if c in df.columns]
    if not cols:
        df['_text_joined'] = ''
        return df
    df['_text_joined'] = df[cols].fillna('').astype(str).agg(' '.join, axis=1)
    return df
