
from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class Paths:
    data_csv: str
    model_dir: str
    reports_dir: str

@dataclass
class Target:
    name: str
    task_type: str  # 'classification'

@dataclass
class Columns:
    text: List[str]
    categorical: List[str]
    datetime: List[str]
    numeric: List[str]

@dataclass
class ModelConfig:
    name: str
    params: dict

@dataclass
class PreprocessingConfig:
    impute_strategy_numeric: str
    impute_strategy_categorical: str
    tfidf_max_features: int
    tfidf_ngram_range: List[int]

@dataclass
class EvaluationConfig:
    test_size: float
    random_state: int
    stratify: bool

@dataclass
class AppConfig:
    paths: Paths
    target: Target
    columns: Columns
    model: ModelConfig
    preprocessing: PreprocessingConfig
    evaluation: EvaluationConfig

def load_config(path: str) -> AppConfig:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return AppConfig(
        paths=Paths(**cfg['paths']),
        target=Target(**cfg['target']),
        columns=Columns(**cfg['columns']),
        model=ModelConfig(**cfg['model']),
        preprocessing=PreprocessingConfig(**cfg['preprocessing']),
        evaluation=EvaluationConfig(**cfg['evaluation']),
    )
