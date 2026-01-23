"""
Monitoring API endpoints for manual triggering and status checks.
"""
from fastapi import APIRouter
from datetime import datetime, timedelta
import psycopg2
import numpy as np
from app.monitoring.runner import run_behavior_monitoring
from app.db.database import get_connection
from app.monitoring.behavior_rules import ENTROPY_THRESHOLD, LOW_CONFIDENCE_RANGE
from app.drift.drift_detection import DriftDetector
from app.drift.reference_loader import load_reference_data
from app.drift.data_loader import load_recent_data

router = APIRouter(prefix="/api", tags=["Monitoring API"])

drift_detector = DriftDetector()


@router.get("/behavior")
def get_behavior_monitoring(model_name: str = "fraud_xgb"):
    """
    Get prediction behavior monitoring data for Page 2.
    Returns baseline vs recent statistics and distribution info.
    """
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # 1. Get baseline stats
        cur.execute("""
            SELECT mean_probability, std_probability, high_risk_ratio
            FROM model_behavior_stats
            WHERE model_name = %s AND window_type = 'BASELINE'
            ORDER BY created_at DESC
            LIMIT 1
        """, (model_name,))
        baseline_row = cur.fetchone()
        
        baseline_stats = {
            "mean_probability": float(baseline_row[0]) if baseline_row else 0,
            "std_probability": float(baseline_row[1]) if baseline_row else 0,
            "high_risk_ratio": float(baseline_row[2]) if baseline_row else 0
        }
        
        # 2. Get recent stats
        cur.execute("""
            SELECT mean_probability, std_probability, high_risk_ratio
            FROM model_behavior_stats
            WHERE model_name = %s AND window_type = 'RECENT'
            ORDER BY created_at DESC
            LIMIT 1
        """, (model_name,))
        recent_row = cur.fetchone()
        
        recent_stats = {
            "mean_probability": float(recent_row[0]) if recent_row else 0,
            "std_probability": float(recent_row[1]) if recent_row else 0,
            "high_risk_ratio": float(recent_row[2]) if recent_row else 0
        }
        
        # 3. Get histogram data (distribution)
        cur.execute("""
            SELECT prediction_probability
            FROM model_predictions
            WHERE model_name = %s AND prediction_probability IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 300
        """, (model_name,))
        
        probs = [float(row[0]) for row in cur.fetchall()]
        
        if probs:
            hist, bins = np.histogram(probs, bins=10)
            histogram_bins = [float(b) for b in bins]
            histogram_counts = [int(c) for c in hist]
        else:
            histogram_bins = []
            histogram_counts = []
        
        # 4. Get time series data (rolling stats over last hour)
        cur.execute("""
            SELECT created_at, prediction_probability
            FROM model_predictions
            WHERE model_name = %s 
              AND prediction_probability IS NOT NULL
              AND created_at > NOW() - INTERVAL '1 hour'
            ORDER BY created_at ASC
        """, (model_name,))
        
        time_series_data = cur.fetchall()
        timestamps = []
        rolling_means = []
        rolling_stds = []
        rolling_risk_ratios = []
        
        if time_series_data:
            window_size = max(1, len(time_series_data) // 10)
            for i in range(0, len(time_series_data), window_size):
                window = [float(row[1]) for row in time_series_data[i:i+window_size]]
                if window:
                    timestamps.append(time_series_data[i][0].isoformat())
                    rolling_means.append(float(np.mean(window)))
                    rolling_stds.append(float(np.std(window)))
                    risk_ratio = float(np.mean(np.array(window) > 0.8))
                    rolling_risk_ratios.append(risk_ratio)
        
        return {
            "model": {
                "name": model_name,
                "version": "v1.0"
            },
            "baseline_stats": baseline_stats,
            "recent_stats": recent_stats,
            "thresholds": {
                "mean_shift_std_multiplier": 1.0,
                "high_risk_upper": baseline_stats["high_risk_ratio"] * 1.5,
                "high_risk_lower": 0.30
            },
            "distribution_snapshot": {
                "histogram_bins": histogram_bins,
                "histogram_counts": histogram_counts
            },
            "time_series": {
                "timestamps": timestamps,
                "rolling_mean": rolling_means,
                "rolling_std": rolling_stds,
                "high_risk_ratio": rolling_risk_ratios
            }
        }
    
    finally:
        cur.close()
        conn.close()


@router.get("/drift")
def get_drift_monitoring(model_name: str = "fraud_xgb"):
    """
    Get data drift monitoring data for Page 3.
    Detects feature-level drift using KS-test.
    """
    try:
        reference_df = load_reference_data()
        recent_df = load_recent_data(limit=300)
        
        drift_report = drift_detector.detect(reference_df, recent_df)
        
        # Extract feature-level details
        feature_drift = []
        if "feature_stats" in drift_report:
            for feature, stats in drift_report["feature_stats"].items():
                feature_drift.append({
                    "feature": feature,
                    "p_value": float(stats.get("p_value", 1.0)),
                    "drift_detected": stats.get("is_drifted", False)
                })
        
        # Calculate summary
        total_features = len(feature_drift)
        drifted_count = sum(1 for f in feature_drift if f["drift_detected"])
        drift_ratio = drifted_count / total_features if total_features > 0 else 0
        
        # Determine status
        if drift_ratio > 0.7:
            status = "HIGH_DRIFT"
        elif drift_ratio > 0.3:
            status = "MODERATE_DRIFT"
        else:
            status = "LOW_DRIFT"
        
        return {
            "drift_summary": {
                "total_features": total_features,
                "drifted_features": drifted_count,
                "drift_ratio": float(drift_ratio),
                "status": status
            },
            "feature_drift": feature_drift,
            "reference_window": {
                "source": "training_data_sample",
                "size": len(reference_df)
            },
            "recent_window": {
                "source": "production_predictions",
                "size": len(recent_df)
            }
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "drift_summary": {
                "total_features": 0,
                "drifted_features": 0,
                "drift_ratio": 0,
                "status": "ERROR"
            },
            "feature_drift": []
        }


@router.get("/alerts")
def get_alerts_monitoring(model_name: str = "fraud_xgb"):
    """
    Get alerts and incidents timeline for Page 4.
    Provides audit trail and alert statistics.
    """
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # 1. Get all alerts from both tables
        alerts = []
        
        # Confidence alerts
        cur.execute("""
            SELECT id, created_at, alert_type, 'CONFIDENCE' as category, 
                   probability, entropy
            FROM model_alerts
            WHERE model_name = %s
            ORDER BY created_at DESC
            LIMIT 100
        """, (model_name,))
        
        for row in cur.fetchall():
            alerts.append({
                "id": row[0],
                "timestamp": row[1].isoformat(),
                "alert_type": row[2],
                "category": row[3],
                "severity": "MEDIUM",
                "message": f"Confidence alert: {row[2]}",
                "status": "ACTIVE"
            })
        
        # Behavior alerts
        cur.execute("""
            SELECT id, created_at, alert_type
            FROM model_behavior_alerts
            WHERE model_name = %s
            ORDER BY created_at DESC
            LIMIT 100
        """, (model_name,))
        
        for row in cur.fetchall():
            alerts.append({
                "id": row[0],
                "timestamp": row[1].isoformat(),
                "alert_type": row[2],
                "category": "MODEL_BEHAVIOR",
                "severity": "HIGH",
                "message": f"Behavior alert: {row[2]}",
                "status": "ACTIVE"
            })
        
        # Sort by timestamp descending
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # 2. Calculate alert statistics
        cur.execute("""
            SELECT COUNT(*) FROM model_alerts
            WHERE model_name = %s AND created_at > NOW() - INTERVAL '24 hours'
        """, (model_name,))
        last_24h = cur.fetchone()[0]
        
        cur.execute("""
            SELECT COUNT(*) FROM model_alerts
            WHERE model_name = %s AND created_at > NOW() - INTERVAL '7 days'
        """, (model_name,))
        last_7d = cur.fetchone()[0]
        
        # 3. Get baseline (last healthy state)
        cur.execute("""
            SELECT mean_probability, high_risk_ratio
            FROM model_behavior_stats
            WHERE model_name = %s AND window_type = 'BASELINE'
            ORDER BY created_at DESC
            LIMIT 1
        """, (model_name,))
        
        baseline_row = cur.fetchone()
        last_healthy = {
            "timestamp": datetime.now().isoformat(),
            "baseline_snapshot": {
                "mean_probability": float(baseline_row[0]) if baseline_row else 0,
                "high_risk_ratio": float(baseline_row[1]) if baseline_row else 0
            }
        }
        
        # Count by type
        by_type = {}
        for alert in alerts:
            alert_type = alert["alert_type"]
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
        
        return {
            "alerts": alerts,
            "alert_statistics": {
                "last_24_hours": last_24h,
                "last_7_days": last_7d,
                "by_type": by_type
            },
            "last_healthy_state": last_healthy
        }
    
    finally:
        cur.close()
        conn.close()


@router.get("/overview")
def get_system_overview(model_name: str = "fraud_xgb"):
    """
    Get system overview for dashboard - Page 1.
    Returns status, last prediction, active alerts, and monitoring coverage.
    """
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


@router.post("/run")
def run_monitoring_manual(model_name: str = "fraud_xgb"):
    """
    Manually trigger behavior monitoring.
    Useful for debugging, testing, or manual checks.
    """
    result = run_behavior_monitoring(model_name)
    return result


@router.get("/status")
def monitoring_status():
    """
    Get current monitoring status (health check).
    """
    return {
        "service": "behavior_monitoring",
        "status": "operational",
        "endpoint": "/dashboard/run"
    }
