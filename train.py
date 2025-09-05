
import os
import argparse
import pandas as pd
from src.config import load_config
from src.data_loading import read_csv, coerce_datetime
from src.modeling import train_pipeline, save_pipeline
from src.utils import save_json

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    df = read_csv(cfg.paths.data_csv)
    df = coerce_datetime(df, cfg.columns.datetime)

    pipe, metrics = train_pipeline(df, cfg)
    model_path = save_pipeline(pipe, cfg.paths.model_dir)
    print(f"Model saved to: {model_path}")
    save_json(metrics, os.path.join(cfg.paths.reports_dir, 'metrics.json'))
    print("Training metrics saved to reports/metrics.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()
    main(args.config)
