import json 
import pandas as pd
import numpy as np

class Preprocessing:
    def __init__(self, schema_path: str):
        with open(schema_path) as f:
            self.schema = json.load(f)
        
        self.drop_features = self.schema["drop_features"]
        self.categorical_features = self.schema["categorical_features"]
        self.encoded_columns = self.schema["encoded_columns"]
        self.final_features_order = self.schema["final_feature_order"]

    def transform(self, raw_input: dict) -> pd.DataFrame:
        df = pd.DataFrame([raw_input])
        
        # Drop unnecessary features first
        df = df.drop(columns=self.drop_features, errors="ignore")
        
        # One-hot encode categorical features
        if self.categorical_features:
            df = pd.get_dummies(df, columns=self.categorical_features, drop_first=False)
        
        # Ensure all encoded columns exist (fill missing categories with 0)
        for col in self.encoded_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Ensure all numeric features in final order exist
        numeric_features = [col for col in self.final_features_order if col not in self.encoded_columns]
        for col in numeric_features:
            if col not in df.columns:
                df[col] = 0
        
        # Select final features in correct order (preserves order and removes extra columns)
        df = df[self.final_features_order]
        
        # Convert to float32 for model compatibility
        df = df.astype(np.float32)
        
        return df