import json
from app.db.database import get_connection

def log_prediction(
        model_name: str,
        model_version: str,
        input_payload: dict,
        prediction: int,
        prediction_probability: float,
        prediction_entropy: float
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """ INSERT INTO model_predictions (
            model_name,
            model_version,
            input_payload,
            fraud_probability,
            prediction,
            prediction_probability,
            prediction_entropy,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """,
        (
            model_name,
            model_version,
            json.dumps(input_payload),
            prediction_probability,
            prediction,
            prediction_probability,
            prediction_entropy
        )
    )

    conn.commit()
    cur.close()
    conn.close()