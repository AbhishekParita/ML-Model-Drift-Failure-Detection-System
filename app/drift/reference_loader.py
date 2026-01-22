import pandas as pd

REFERENCE_PATH = "app/models/reference_data.csv"

def load_reference_data():
    """Load reference baseline data from CSV file"""
    ref_df = pd.read_csv(REFERENCE_PATH)
    
    if ref_df.empty:
        raise RuntimeError("Reference dataset is empty")
    
    return ref_df