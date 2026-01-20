import json
import pandas as pd
import numpy as np

def compute_reference_stats(df: pd.DataFrame):
    """Compute reference statistics from training data (baseline)"""
    stats = {}
    
    # Convert all columns to float to handle boolean/int encoded columns
    df = df.astype(float)

    for col in df.columns:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median()),
            'q10': float(df[col].quantile(0.1)),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75)),
            'q90': float(df[col].quantile(0.9)),
            'count': int(df[col].count())
        }

    return stats

def save_reference_stats(stats: dict, path: str):
    """Save reference stats to JSON file"""
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Reference stats saved to {path}")

def load_reference_stats(path: str):
    """Load reference stats from JSON file"""
    with open(path, 'r') as f:
        stats = json.load(f)
    print(f"✅ Reference stats loaded from {path}")
    return stats