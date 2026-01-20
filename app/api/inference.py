import joblib 
from app.core.preprocessing import Preprocessing
from app.db.logger import log_prediction

MODEL_PATH = "app\models/base_model.pkl"
SCHEMA_PATH = "app\models/feature_schema.json"

MODEL_NAME = "fraud_xgb"
MODEL_VERSION = "v1.0"

model = joblib.load(MODEL_PATH)
preprocessor = Preprocessing(SCHEMA_PATH)

def predict_fraud(input_data: dict) -> float:
    """Predict fraud probability, log prediction, and return as Python float for JSON serialization"""
    X = preprocessor.transform(input_data)
    prob = model.predict_proba(X)[0][1]
    predicted_label = int(prob >= 0.5)
    
    # Log prediction to database (wrapped in try-except to prevent crashes)
    try:
        log_prediction(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            input_payload=input_data,
            fraud_probability=float(prob),
            predicted_label=predicted_label
        )
    except Exception as e:
        print(f"Warning: Failed to log prediction - {str(e)}")
    
    # Convert numpy.float32 to Python float for JSON serialization
    return float(prob)
