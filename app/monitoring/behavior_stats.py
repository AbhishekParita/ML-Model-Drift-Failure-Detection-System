import numpy as np

def compute_behavior_stats(probabilities: list[float]):
    probs = np.array(probabilities)

    return {
        "mean_probabilities": float(np.mean(probs)),
        "std_probability": float(np.std(probs)),
        "high_risk_ration": float(np.mean(probs > 0.8))
    }

