import joblib 
from app.core.preprocessing import Preprocessing

MODEL_PATH = "app\models/base_model.pkl"
SCHEMA_PATH = "app\models/feature_schema.json"

model = joblib.load(MODEL_PATH)
preprocessor = Preprocessing(SCHEMA_PATH)

def predict_fraud(input_data: dict) -> float:
    """Predict fraud probability and return as Python float for JSON serialization"""
    X = preprocessor.transform(input_data)
    prob = model.predict_proba(X)[0][1]
    # Convert numpy.float32 to Python float
    return float(prob)
