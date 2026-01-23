import joblib
import numpy as np
from app.core.preprocessing import Preprocessor
from app.db.logger import log_prediction
from app.db.alert_logger import log_alert
from app.monitoring.behaviour_monitor import evaluate_behaviour

MODEL_PATH = "app/models/base_model.pkl"
SCHEMA_PATH = "app/models/feature_schema.json"

MODEL_NAME = "fraud_xgb"
MODEL_VERSION = "v1.0"

model = joblib.load(MODEL_PATH)
preprocessor = Preprocessor(SCHEMA_PATH)

def predict_fraud(input_data: dict) -> float:
    """Predict fraud probability, calculate entropy, log prediction, and return as Python float for JSON serialization"""
    X = preprocessor.transform(input_data)
    
    # Compute variables BEFORE insert
    p = float(model.predict_proba(X)[0][1])
    prediction = int(p >= 0.5)
    
    entropy = -(
        p * np.log(p + 1e-9) +
        (1 - p) * np.log(1 - p + 1e-9)
    )
    entropy = float(entropy)
    
    # Log prediction to database - fail explicitly if logging fails
    try:
        log_prediction(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            input_payload=input_data,
            prediction=prediction,
            prediction_probability=p,
            prediction_entropy=entropy
        )
    except Exception as e:
        raise RuntimeError(f"Prediction logging failed: {e}")
    
    # Evaluate behavior and log alerts
    alerts = evaluate_behaviour(p, entropy)
    for alert_type in alerts:
        try:
            log_alert(
                model_name=MODEL_NAME,
                alert_type=alert_type,
                probability=p,
                entropy=entropy
            )
        except Exception as e:
            # Log alert failures but don't fail the prediction
            print(f"Warning: Alert logging failed for {alert_type}: {e}")
    
    # Return fraud probability
    return p
