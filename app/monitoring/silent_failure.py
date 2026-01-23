import numpy as np
from app.db.database import get_connection


def detect_silent_shift(baseline, recent):
    """
    Detect behavioral drift by comparing recent stats to baseline.
    
    Args:
        baseline: dict with keys ["mean", "std", "high_risk_ratio"]
        recent: dict with keys ["mean", "std", "high_risk_ratio"]
    
    Returns:
        str: Alert type ("MEAN_SHIFT", "RISK_RATIO_SHIFT") or None
    """
    # Check 1: Mean probability shifted beyond 1 standard deviation
    if abs(recent["mean"] - baseline["mean"]) > baseline["std"]:
        return "MEAN_SHIFT"
    
    # Check 2: High-risk ratio shifted (hybrid approach)
    # Upper bound: dynamic (baseline × 1.5)
    # Lower bound: fixed (0.30)
    upper_threshold = baseline["high_risk_ratio"] * 1.5
    lower_threshold = 0.30
    
    if recent["high_risk_ratio"] > upper_threshold or recent["high_risk_ratio"] < lower_threshold:
        return "RISK_RATIO_SHIFT"
    
    return None


def log_behavior_alert(model_name: str, alert_type: str, baseline_stats: dict, recent_stats: dict):
    """
    Log behavioral drift alert to database.
    
    Args:
        model_name: str (e.g., 'fraud_xgb')
        alert_type: str ('MEAN_SHIFT' or 'RISK_RATIO_SHIFT')
        baseline_stats: dict with baseline statistics
        recent_stats: dict with recent statistics
    """
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute(
        """ INSERT INTO model_behavior_alerts (
            model_name,
            alert_type,
            baseline_mean,
            recent_mean,
            baseline_high_risk_ratio,
            recent_high_risk_ratio
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            model_name,
            alert_type,
            float(baseline_stats["mean"]),
            float(recent_stats["mean"]),
            float(baseline_stats["high_risk_ratio"]),
            float(recent_stats["high_risk_ratio"])
        )
    )
    
    conn.commit()
    cur.close()
    conn.close()