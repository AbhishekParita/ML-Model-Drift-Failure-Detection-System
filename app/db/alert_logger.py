"""Alert logging to model_alerts table"""
import numpy as np
from app.db.database import get_connection


def log_alert(
    model_name: str,
    alert_type: str,
    probability: float,
    entropy: float
):
    """Log prediction alert to database"""
    # Convert numpy types to native Python floats
    probability = float(probability) if isinstance(probability, (np.floating, np.ndarray)) else probability
    entropy = float(entropy) if isinstance(entropy, (np.floating, np.ndarray)) else entropy
    
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """ INSERT INTO model_alerts (
                model_name,
                alert_type,
                probability,
                entropy,
                created_at
            )
            VALUES (%s, %s, %s, %s, NOW())
            """,
            (
                model_name,
                alert_type,
                probability,
                entropy
            )
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Alert logging failed: {e}")
    finally:
        cur.close()
        conn.close()
