
import os
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .features import build_feature_processor
from .preprocessing import split_X_y
from .nlp_text import join_text_columns

def build_model(cfg) -> Pipeline:
    if cfg.model.name == 'random_forest':
        estimator = RandomForestClassifier(**cfg.model.params)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")
    return estimator

def train_pipeline(df: pd.DataFrame, cfg) -> Tuple[Pipeline, dict]:
    # Prepare data
    X, y = split_X_y(df, cfg.target.name)
    X = X.copy()
    X = join_text_columns(X, cfg.columns.text)

    preprocessor = build_feature_processor(
        cfg.columns.numeric, cfg.columns.categorical, cfg.columns.text, cfg
    )
    estimator = build_model(cfg)

    pipe = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', estimator)
    ])

    stratify = y if (y is not None and cfg.evaluation.stratify) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.evaluation.test_size, random_state=cfg.evaluation.random_state, stratify=stratify
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return pipe, metrics

def save_pipeline(pipe, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(pipe, path)
    return path

def load_pipeline(model_dir: str):
    path = os.path.join(model_dir, 'model.joblib')
    return joblib.load(path)
