# Hard alert thresholds - no ML, just rule-based logic
ENTROPY_THRESHOLD = 0.45
LOW_CONFIDENCE_RANGE = (0.4, 0.6)

# Alert type definitions
ALERT_TYPES = {
    "HIGH_PREDICTION_ENTROPY": "Model uncertainty exceeds safe threshold",
    "LOW_MODEL_CONFIDENCE": "Prediction probability in borderline range (40-60%)",
}
