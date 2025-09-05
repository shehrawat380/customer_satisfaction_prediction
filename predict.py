
import os
import argparse
import pandas as pd
from src.config import load_config
from src.data_loading import read_csv, coerce_datetime
from src.modeling import load_pipeline
from src.nlp_text import join_text_columns

def main(cfg_path: str, input_path: str, output_path: str):
    cfg = load_config(cfg_path)
    df = read_csv(input_path)
    df = coerce_datetime(df, cfg.columns.datetime)
    X = df.copy()
    X = join_text_columns(X, cfg.columns.text)

    pipe = load_pipeline(cfg.paths.model_dir)
    preds = pipe.predict(X)
    out = df.copy()
    out['Predicted Satisfaction'] = preds
    out.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='predictions.csv')
    args = parser.parse_args()
    main(args.config, args.input, args.output)
