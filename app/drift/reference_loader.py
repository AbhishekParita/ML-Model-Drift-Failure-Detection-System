import pandas as pd

def load_reference_data(path="app/models/reference_data.parquet"):
    return pd.read_parquet(path)