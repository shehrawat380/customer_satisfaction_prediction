
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

def build_feature_processor(numeric_cols, categorical_cols, text_cols, cfg):
    numeric = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.impute_strategy_numeric))
    ])

    categorical = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.impute_strategy_categorical)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine all text fields into one by space-joining them inside a custom transformer
    # We'll implement this by using 'TfidfVectorizer' on a joined text column upstream.
    # Here, we simply place a placeholder; actual joining happens in train/inference.
    text = TfidfVectorizer(
        max_features=cfg.preprocessing.tfidf_max_features,
        ngram_range=tuple(cfg.preprocessing.tfidf_ngram_range)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric, numeric_cols),
            ('cat', categorical, categorical_cols),
            # Text is passed as a single combined column named '_text_joined'
            ('txt', text, '_text_joined')
        ],
        remainder='drop',
        sparse_threshold=0.3
    )
    return preprocessor
