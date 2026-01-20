import json
from app.db.database import get_connection

def log_prediction(
        model_name: str,
        model_version: str,
        input_payload: dict,
        fraud_probability: float,
        predicted_label: int= None
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """ INSERT INTO model_predictions
        (model_name, model_version, input_payload, fraud_probability, predicted_label)
        VALUES(%s,%s,%s,%s,%s)
        """,
        (
            model_name,
            model_version,
            json.dumps(input_payload),
            fraud_probability,
            predicted_label
        )
    )

    conn.commit()
    cur.close()
    conn.close()