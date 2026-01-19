from fastapi import FastAPI
from pydantic import BaseModel
from app.api.inference import predict_fraud

app = FastAPI(title="ML Drift Detection System", version="1.0.0")

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

@app.get("/health")
def health_check():
    return {"status": "healthy"}