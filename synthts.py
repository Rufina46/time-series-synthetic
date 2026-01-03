"""
Synthetic Time-Series Generator using SDV 1.x (with metadata)
- Input: CSV with 'timestamp' and numeric/categorical columns
- Output: Synthetic CSV with same schema
- Author: [Your Name] • https://github.com/yourname/synthts
"""

import argparse
import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import warnings
warnings.filterwarnings("ignore")

def load_data(filepath):
    df = pd.read_csv(filepath)
    if 'timestamp' not in df.columns:
        raise ValueError("Input CSV must contain a 'timestamp' column")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def generate_synthetic(real_df, n_samples=50):
    # Подготовка данных
    cols_to_drop = ['timestamp']
    if 'id' in real_df.columns:
        cols_to_drop.append('id')
    data_for_synthesis = real_df.drop(columns=cols_to_drop)

    # Создаём metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data_for_synthesis)

    # Инициализируем синтезатор с metadata
    synthesizer = GaussianCopulaSynthesizer(metadata)
    print("Training synthesizer... (≈1–2 min)")
    synthesizer.fit(data_for_synthesis)

    print("Generating synthetic data...")
    synth_data = synthesizer.sample(num_rows=n_samples)

    return synth_data

def add_back_columns(synth_data, original_df):
    # добавляем идентификатор синтетической строки
    synth_data = synth_data.copy()
    synth_data['synth_id'] = [f's_{i}' for i in range(len(synth_data))]
    return synth_data

def evaluate_quality(real_df, synth_df):
    # используем numpy напрямую (pd.np удалён в новых версиях pandas)
    real_vals = real_df.select_dtypes(include=[np.number]).values.flatten()
    synth_vals = synth_df.select_dtypes(include=[np.number]).values.flatten()
    print("\n[Quality Report]")
    print(f"Real   — mean: {real_vals.mean():.3f}, std: {real_vals.std():.3f}")
    print(f"Synth  — mean: {synth_vals.mean():.3f}, std: {synth_vals.std():.3f}")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic time-series data')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n_samples', type=int, default=50)
    args = parser.parse_args()

    df = load_data(args.input)
    print(f"✅ Loaded {len(df)} rows from {args.input}")

    synth = generate_synthetic(df, n_samples=args.n_samples)
    synth_with_id = add_back_columns(synth, df)
    synth_with_id.to_csv(args.output, index=False)
    print(f"✅ Saved synthetic data to {args.output}")

    evaluate_quality(df, synth_with_id)

if __name__ == "__main__":
    main()