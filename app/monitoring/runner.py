"""
Core monitoring pipeline runner.
Fetches baseline, computes recent stats, detects silent shifts, logs alerts.
"""
import numpy as np
from app.db.database import get_connection
from app.monitoring.behavior_stats import compute_behavior_stats
from app.monitoring.silent_failure import detect_silent_shift, log_behavior_alert

RECENT_WINDOW = 50  # Number of recent predictions to analyze


def run_behavior_monitoring(model_name: str = "fraud_xgb"):
    """
    Run the behavior monitoring pipeline.
    
    Args:
        model_name: str (e.g., 'fraud_xgb')
    
    Returns:
        dict with status and reason
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Step 1: Load baseline stats
        cur.execute("""
            SELECT mean_probability, std_probability, high_risk_ratio
            FROM model_behavior_stats
            WHERE model_name = %s AND window_type = 'BASELINE'
            ORDER BY created_at DESC
            LIMIT 1
        """, (model_name,))
        
        baseline_row = cur.fetchone()
        
        if not baseline_row:
            cur.close()
            conn.close()
            return {
                "status": "skipped",
                "reason": "baseline_not_found",
                "model": model_name
            }
        
        baseline_stats = {
            "mean": float(baseline_row[0]),
            "std": float(baseline_row[1]),
            "high_risk_ratio": float(baseline_row[2]),
        }
        
        # Step 2: Load recent prediction probabilities
        cur.execute("""
            SELECT prediction_probability
            FROM model_predictions
            WHERE model_name = %s
              AND prediction_probability IS NOT NULL
            ORDER BY created_at DESC
            LIMIT %s
        """, (model_name, RECENT_WINDOW))
        
        rows = cur.fetchall()
        
        if len(rows) < 10:
            cur.close()
            conn.close()
            return {
                "status": "skipped",
                "reason": "not_enough_data",
                "data_count": len(rows),
                "required": 10
            }
        
        probs = [float(r[0]) for r in rows]
        recent_stats = compute_behavior_stats(probs)
        
        # Step 3: Detect silent shift
        alert_type = detect_silent_shift(baseline_stats, recent_stats)
        
        if alert_type:
            # Insert alert
            log_behavior_alert(
                model_name=model_name,
                alert_type=alert_type,
                baseline_stats=baseline_stats,
                recent_stats=recent_stats
            )
            
            cur.close()
            conn.close()
            
            return {
                "status": "alert_triggered",
                "reason": alert_type,
                "model": model_name,
                "baseline": baseline_stats,
                "recent": recent_stats
            }
        
        cur.close()
        conn.close()
        
        return {
            "status": "healthy",
            "model": model_name,
            "baseline": baseline_stats,
            "recent": recent_stats
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "model": model_name
        }
