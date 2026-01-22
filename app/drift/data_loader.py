import pandas as pd
from app.db.database import get_connection
from app.core.preprocessing import Preprocessor

preprocessor = Preprocessor("app/models/feature_schema.json")

def load_recent_data(limit=500):
    """Load recent prediction data from database and preprocess"""
    conn = get_connection()

    query = """
    SELECT input_payload
    FROM model_predictions
    ORDER BY created_at DESC
    LIMIT %s
    """

    df = pd.read_sql(query, conn, params=(limit,))
    conn.close()

    if df.empty:
        raise RuntimeError("No live data fetched from database")

    raw_df = pd.json_normalize(df["input_payload"])

    processed_rows = []
    for _, row in raw_df.iterrows():
        processed = preprocessor.transform(row.to_dict())
        processed_rows.append(processed)

    live_df = pd.concat(processed_rows, ignore_index=True)
    return live_df
