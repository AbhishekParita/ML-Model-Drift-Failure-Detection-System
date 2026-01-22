from app.monitoring.behavior_rules import ENTROPY_THRESHOLD, LOW_CONFIDENCE_RANGE

def evaluate_behaviour(probability: float, entropy: float):
    """Evaluate prediction behavior against hard alert rules"""
    alerts = []
    
    # Rule 1: High entropy indicates model uncertainty
    if entropy > ENTROPY_THRESHOLD:
        alerts.append("HIGH_PREDICTION_ENTROPY")
    
    # Rule 2: Low confidence in borderline range
    if LOW_CONFIDENCE_RANGE[0] <= probability <= LOW_CONFIDENCE_RANGE[1]:
        alerts.append("LOW_MODEL_CONFIDENCE")

    return alerts