from scipy.stats import ks_2samp

class DriftDetector:
    """Detects data drift using Kolmogorov-Smirnov test"""
    
    def __init__(self, p_value_threshold=0.05):
        """
        Initialize drift detector
        
        Args:
            p_value_threshold: Significance level for KS test (default: 0.05)
        """
        self.threshold = p_value_threshold

    def detect(self, reference_df, live_df):
        """
        Detect drift by comparing reference and live data distributions
        
        Args:
            reference_df: Reference/baseline data
            live_df: Recent/live data to check for drift
            
        Returns:
            Dictionary with per-feature p-values and drift detection results
        """
        report = {}
        global_drift_count = 0

        for col in reference_df.columns:
            stat, p_value = ks_2samp(
                reference_df[col],
                live_df[col]
            )

            drift = p_value < self.threshold
            if drift:
                global_drift_count += 1

            report[col] = {
                "p_value": float(p_value),
                "drift_detected": drift
            }

        report["summary"] = {
            "total_features": len(reference_df.columns),
            "drifted_features": global_drift_count,
            "drift_ratio": global_drift_count / len(reference_df.columns)
        }

        return report