from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from app.api.inference import predict_fraud
from app.drift.data_loader import load_recent_data
from app.drift.reference_loader import load_reference_data
from app.drift.drift_detection import DriftDetector

app = FastAPI(title="ML Drift Detection System", version="1.0.0")

# Initialize drift detector
drift_detector = DriftDetector()

class PredictionRequest(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int
    type: str

class PredictionResponse(BaseModel):
    fraud_probability: float
    success: bool = True

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    """Predict fraud probability for given transaction"""
    input_dict = data.dict()
    prob = predict_fraud(input_dict)
    return {"fraud_probability": prob, "success": True}

@app.get("/drift")
def check_drift(limit: int = 500):
    """Detect data drift by comparing recent predictions with reference baseline"""
    try:
        # Load reference baseline data
        reference_df = load_reference_data()
        
        # Load recent prediction data from database
        live_df = load_recent_data(limit=limit)
        
        # Detect drift using improved detector
        drift_report = drift_detector.detect(reference_df, live_df)
        
        return drift_report
    except Exception as e:
        return {
            "error": str(e),
            "summary": {
                "total_features": 0,
                "drifted_features": 0,
                "drift_ratio": 0
            }
        }

@app.get("/health")
def health_check():
    return {"status": "healthy"}