"""
Sensitivity-weighted NCP computation.
"""
from typing import Dict, Sequence


def compute_sensitivity_weights(
    feature_names: Sequence[str],
    sensitive_features: Sequence[str],
    alpha: float = 2.0,
) -> Dict[str, float]:
    """Return per-feature weights for the weighted NCP variant.

    Weights are rescaled to sum to ``len(feature_names)`` so the weighted
    average stays on the same scale as the original uniform NCP.

    :param feature_names: All feature names in the dataset.
    :param sensitive_features: Subset to up-weight.
    :param alpha: Boost factor; sensitive weight = 1 + alpha. 0 gives uniform.
    :return: ``{feature: normalised_weight}`` dict.
    """
    feature_names = list(feature_names)
    sensitive_set = set(sensitive_features)
    n = len(feature_names)
    raw = {
        f: (1.0 + float(alpha)) if f in sensitive_set else 1.0
        for f in feature_names
    }
    # Keep weighted NCP on the original scale by normalizing weight sum to n
    scale = float(n) / sum(raw.values())
    return {f: raw[f] * scale for f in feature_names}
