import numpy as np

def compute_behavior_stats(probabilities: list[float]):
    """
    Compute behavioral statistics from prediction probabilities.
    
    Args:
        probabilities: list of fraud probabilities
    
    Returns:
        dict with keys: mean, std, high_risk_ratio
    """
    probs = np.array(probabilities)

    return {
        "mean": float(np.mean(probs)),
        "std": float(np.std(probs)),
        "high_risk_ratio": float(np.mean(probs > 0.8))
    }

