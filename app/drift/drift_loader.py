import pandas as pd
from app.db.database import get_connection

def load_recent(limit = 500):
    conn = get_connection()

    query = """ 
    SELECT input_payload
    FROM model_predictions
    ORDER BY created_at DESC
    LIMIT %s
    """

    df = pd.read_sql(query, conn, params = (limit,))
    conn.close()

    features_df = pd.json_normalize(df["input_payload"])
    return features_df