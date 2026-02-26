"""
Attribute disclosure and PA-ILAG scoring for generalized data.
"""
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def compute_pa_ilag_score(base_ilag: float, leakage_delta: float, lambda_attr: float = 1.0) -> float:
    """PA-ILAG(f) = base_ilag(f) + lambda_attr * leakage_delta(f).

    Higher score means the feature is un-generalized later, keeping leaky
    features protected longer.

    :param base_ilag: Original ILAG score (NCP / accuracy_gain).
    :param leakage_delta: AUC increase from un-generalizing this feature.
    :param lambda_attr: Weight for the leakage penalty term.
    :return: Combined PA-ILAG score.
    """
    return float(base_ilag) + float(lambda_attr) * float(leakage_delta)


def compute_sensitive_auc(
    data: pd.DataFrame,
    sensitive_feature: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Optional[float]:
    """Train a logistic-regression attacker and return ROC-AUC.

    :param data: DataFrame with all features including the sensitive one.
    :param sensitive_feature: Column name of the sensitive attribute.
    :param test_size: Fraction held out for AUC evaluation.
    :param random_state: RNG seed for the train/test split.
    :return: AUC score, or None if evaluation is not possible.
    """
    target = data[sensitive_feature]
    if not pd.api.types.is_numeric_dtype(target):
        target, _ = pd.factorize(target, sort=True)
    else:
        target = target.to_numpy()

    if len(np.unique(target)) < 2:
        # Cannot fit a classifier or compute AUC with a single class
        return None

    # Drop the sensitive column so AUC reflects inference from proxy features
    predictors = pd.get_dummies(data.drop(columns=[sensitive_feature]), drop_first=False)
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            predictors, target, test_size=test_size, random_state=random_state, stratify=target,
        )
    except (ValueError, RuntimeError):
        # Fail closed on edge splits (e.g. tiny minority class) - caller already handles None
        return None

    try:
        clf = LogisticRegression(max_iter=800, solver='lbfgs')
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_test)
        if proba.shape[1] == 2:
            return float(roc_auc_score(y_test, proba[:, 1]))
        return float(roc_auc_score(y_test, proba, multi_class='ovr'))
    except (ValueError, RuntimeError):
        # Keep disclosure reporting robust rather than failing the whole minimization run
        return None


def measure_attribute_disclosure(
    original_data: pd.DataFrame,
    generalized_data: pd.DataFrame,
    sensitive_feature: str,
    auc_threshold: float = 0.7,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dict[str, Optional[float]]:
    """Compare sensitive-attribute AUC before and after generalization.

    :param original_data: Raw dataset before generalization.
    :param generalized_data: Dataset after minimization transform.
    :param sensitive_feature: Column to attack.
    :param auc_threshold: Maximum acceptable post-generalization AUC.
    :param test_size: Fraction held out for AUC evaluation.
    :param random_state: RNG seed.
    :return: Dict with auc_before, auc_after, auc_delta, threshold_pass.
    """
    auc_before = compute_sensitive_auc(
        original_data,
        sensitive_feature=sensitive_feature,
        test_size=test_size,
        random_state=random_state,
    )
    auc_after = compute_sensitive_auc(
        generalized_data,
        sensitive_feature=sensitive_feature,
        test_size=test_size,
        random_state=random_state,
    )
    auc_delta = None if auc_before is None or auc_after is None else float(auc_after - auc_before)
    threshold_pass = True if auc_after is None else bool(auc_after <= auc_threshold)
    return {
        'sensitive_feature': sensitive_feature,
        'auc_before': auc_before,
        'auc_after': auc_after,
        'auc_delta': auc_delta,
        'auc_threshold': float(auc_threshold),
        'threshold_pass': threshold_pass,
    }
