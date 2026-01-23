from fastapi import FastAPI, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
from app.api.inference import predict_fraud
from app.api.monitoring import router as monitoring_router
from app.drift.data_loader import load_recent_data
from app.drift.reference_loader import load_reference_data
from app.drift.drift_detection import DriftDetector
from app.monitoring.runner import run_behavior_monitoring
import requests

app = FastAPI(title="ML Drift Detection System", version="1.0.0")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Register monitoring router
app.include_router(monitoring_router)

# Initialize drift detector
drift_detector = DriftDetector()

class PredictionRequest(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: str

class PredictionResponse(BaseModel):
    fraud_probability: float
    success: bool = True

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest, background_tasks: BackgroundTasks):
    """Predict fraud probability for given transaction"""
    input_dict = data.dict()
    prob = predict_fraud(input_dict)
    
    # Add monitoring task to background (async, doesn't block response)
    background_tasks.add_task(run_behavior_monitoring)
    
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


@app.get("/")
def dashboard_overview(request: Request):
    """
    Render dashboard Page 1: System Overview
    Fetches data from /monitor/overview endpoint
    """
    try:
        # Call the API endpoint internally
        overview_data = get_system_overview()
        
        return templates.TemplateResponse(
            "overview.html",
            {
                "request": request,
                "page": "overview",
                "data": overview_data,
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
    except Exception as e:
        return {
            "error": f"Dashboard error: {str(e)}",
            "page": "overview"
        }


@app.get("/dashboard/behavior")
def dashboard_behavior(request: Request):
    """
    Render dashboard Page 2: Behavior Monitoring
    """
    try:
        # Call monitoring router endpoint
        response = requests.get("http://127.0.0.1:8000/api/behavior")
        behavior_data = response.json()
        
        return templates.TemplateResponse(
            "behavior.html",
            {
                "request": request,
                "page": "behavior",
                "data": behavior_data,
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
    except Exception as e:
        return {
            "error": f"Behavior monitoring error: {str(e)}",
            "page": "behavior"
        }


@app.get("/dashboard/drift")
def dashboard_drift(request: Request):
    """
    Render dashboard Page 3: Data Drift Monitoring
    """
    try:
        # Call monitoring router endpoint
        response = requests.get("http://127.0.0.1:8000/api/drift")
        drift_data = response.json()
        
        return templates.TemplateResponse(
            "drift.html",
            {
                "request": request,
                "page": "drift",
                "data": drift_data,
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
    except Exception as e:
        return {
            "error": f"Drift monitoring error: {str(e)}",
            "page": "drift"
        }


@app.get("/dashboard/alerts")
def dashboard_alerts(request: Request):
    """
    Render dashboard Page 4: Alerts Timeline
    """
    try:
        # Call monitoring router endpoint
        response = requests.get("http://127.0.0.1:8000/api/alerts")
        alerts_data = response.json()
        
        return templates.TemplateResponse(
            "alerts.html",
            {
                "request": request,
                "page": "alerts",
                "data": alerts_data,
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
    except Exception as e:
        return {
            "error": f"Alerts monitoring error: {str(e)}",
            "page": "alerts"
        }


def get_system_overview(model_name: str = "fraud_xgb"):
    """Internal helper to fetch overview data"""
    from app.db.database import get_connection
    
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # 1. Get last prediction
        cur.execute("""
            SELECT created_at, prediction_probability, prediction_entropy
            FROM model_predictions
            WHERE model_name = %s AND prediction_probability IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
        """, (model_name,))
        
        last_pred = cur.fetchone()
        last_prediction = {
            "timestamp": last_pred[0].isoformat() if last_pred else None,
            "fraud_probability": float(last_pred[1]) if last_pred else None,
            "prediction_entropy": float(last_pred[2]) if last_pred else None
        } if last_pred else None
        
        # 2. Count active alerts
        cur.execute("""
            SELECT COUNT(*) 
            FROM model_alerts 
            WHERE model_name = %s 
            AND alert_type IN ('HIGH_PREDICTION_ENTROPY', 'LOW_MODEL_CONFIDENCE')
        """, (model_name,))
        confidence_alerts = cur.fetchone()[0]
        
        cur.execute("""
            SELECT COUNT(*) 
            FROM model_behavior_alerts 
            WHERE model_name = %s
        """, (model_name,))
        behavior_alerts = cur.fetchone()[0]
        
        cur.execute("""
            SELECT COUNT(*) 
            FROM model_predictions
            WHERE model_name = %s
        """, (model_name,))
        total_predictions = cur.fetchone()[0]
        
        data_drift_alerts = 0
        total_alerts = confidence_alerts + behavior_alerts + data_drift_alerts
        
        # 3. Determine status
        if total_alerts >= 3:
            status = "CRITICAL"
        elif total_alerts >= 1:
            status = "WARNING"
        else:
            status = "HEALTHY"
        
        # 4. Monitoring coverage
        monitoring_coverage = {
            "data_drift": True,
            "silent_failure": True,
            "background_monitoring": True
        }
        
        return {
            "model": {
                "name": model_name,
                "version": "v1.0"
            },
            "status": status,
            "last_prediction": last_prediction,
            "active_alerts": {
                "data_drift": data_drift_alerts,
                "behavior_shift": behavior_alerts,
                "confidence": confidence_alerts,
                "total": total_alerts
            },
            "monitoring_coverage": monitoring_coverage,
            "total_predictions_logged": total_predictions
        }
    
    finally:
        cur.close()
        conn.close()