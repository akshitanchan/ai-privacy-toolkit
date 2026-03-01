import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from apt.minimization import GeneralizeToRepresentative
from apt.minimization.security_metrics import compute_pa_ilag_score, compute_sensitive_auc
from apt.minimization.security_postprocess import (
    _compute_emd,
    _compute_variational_distance,
    measure_tcloseness,
    randomize_cell_representatives,
)
from apt.minimization.weighted_ncp import compute_sensitivity_weights
from apt.utils.datasets import ArrayDataset


# Build a small synthetic tabular dataset with one sensitive binary column for use across tests
def _make_dataset(seed=9, n=80):
    rng = np.random.RandomState(seed)
    age = rng.randint(18, 70, size=n)
    hours = rng.randint(10, 70, size=n)
    sensitive = (age > 42).astype(int)
    score = age + hours // 2 + sensitive * 3
    labels = (score > np.median(score)).astype(int)
    return pd.DataFrame({'age': age, 'hours': hours, 'sensitive': sensitive}), labels


# PA-ILAG formula
def test_pa_ilag_score():
    # Higher leakage raises the score (feature un-generalized later)
    assert compute_pa_ilag_score(0.25, 0.01) < compute_pa_ilag_score(0.25, 0.20)

    # lambda=0 disables the penalty: score equals raw base_ilag
    assert compute_pa_ilag_score(0.05, 0.40, lambda_attr=0.0) == 0.05

    # Proxy feature A scores higher than safe feature B so B is selected first
    assert compute_pa_ilag_score(0.05, 0.40) > compute_pa_ilag_score(0.20, 0.00)


# Sensitivity-weighted NCP
def test_sensitivity_weights():
    names = ['age', 'sex', 'hours', 'income']
    w = compute_sensitivity_weights(names, ['sex'], alpha=2.0)

    # Sensitive feature must outweigh every non-sensitive feature
    assert all(w['sex'] > w[f] for f in names if f != 'sex')

    # Weights sum to n (preserves NCP scale)
    assert abs(sum(w.values()) - float(len(names))) < 1e-9

    # alpha=0 yields uniform weights of 1.0
    u = compute_sensitivity_weights(names, ['sex'], alpha=0.0)
    assert all(abs(u[f] - 1.0) < 1e-9 for f in names)


# DP-inspired representative randomization
def test_dp_representatives_non_verbatim():
    data = pd.DataFrame({
        'age': [21, 22, 41, 42], 'income': [100, 120, 400, 410], 'group': ['a', 'a', 'b', 'b'],
    })
    cells = [
        {'id': 1, 'label': [0],
         'ranges': {'age': {'start': None, 'end': 30}, 'income': {'start': None, 'end': 200}},
         'categories': {'group': ['a']}, 'untouched': [],
         'representative': {'age': 21, 'income': 100, 'group': 'a'}},
        {'id': 2, 'label': [1],
         'ranges': {'age': {'start': 30, 'end': None}, 'income': {'start': 200, 'end': None}},
         'categories': {'group': ['b']}, 'untouched': [],
         'representative': {'age': 41, 'income': 400, 'group': 'b'}},
    ]
    feat_data = {
        'age': {'min': 21, 'max': 42, 'range': 21},
        'income': {'min': 100, 'max': 410, 'range': 310},
        'group': {'range': 2},
    }

    # max_retries=0 forces the deterministic fallback path for both cells
    rpt = randomize_cell_representatives(
        cells, data, ['age', 'income', 'group'], feat_data,
        epsilon=1.0, max_retries=0, random_state=7,
    )
    assert rpt['fallback_count'] == 2
    for cell in cells:
        mask = np.ones(len(data), dtype=bool)
        for f, v in cell['representative'].items():
            mask &= (data[f] == v).to_numpy()
        assert not mask.any(), 'representative should not match any training row'


# t-closeness distance functions and integrator
def test_tcloseness():
    p = np.array([0.5, 0.5])
    assert _compute_variational_distance(p, p) == 0.0
    assert _compute_emd(p, p) == 0.0

    # Maximum variational distance between opposite distributions
    assert abs(_compute_variational_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) - 1.0) < 1e-9

    # Toy dataset with cell 1 = all M, cell 2 = all F and global at 50/50
    data = pd.DataFrame({'age': [20, 25, 40, 45], 'sex': ['M', 'M', 'F', 'F']})
    cells = [
        {'id': 1, 'ranges': {'age': {'start': None, 'end': 30}}, 'categories': {}, 'untouched': ['sex']},
        {'id': 2, 'ranges': {'age': {'start': 30, 'end': None}}, 'categories': {}, 'untouched': ['sex']},
    ]
    rpt = measure_tcloseness(cells, data, ['age', 'sex'], ['sex'], t_threshold=0.3)
    assert abs(rpt['per_feature']['sex']['max_distance'] - 0.5) < 1e-9
    assert rpt['per_feature']['sex']['cells_violating_t'] == 2


# AUC helper robustness
def test_sensitive_auc_none_on_failure():
    data = pd.DataFrame({'proxy': [0, 1, 2, 3], 'sensitive': [0, 0, 0, 1]})
    assert compute_sensitive_auc(data, 'sensitive') is None


# Divergence tracking (fit-level, Feature 1 isolation)
def test_divergence_nonzero_with_weights():
    # With strong sensitivity weights and lambda=0, weighted ILAG order must differ from raw ILAG in atleast one step
    x, labels = _make_dataset()
    feat = x.columns.tolist()
    est = DecisionTreeClassifier(random_state=0, min_samples_split=2, min_samples_leaf=1)
    est.fit(x, labels)
    minimizer = GeneralizeToRepresentative(
        estimator=est, target_accuracy=0.99, features_to_minimize=feat,
        weighted_ncp_config={'sensitive_features': ['sensitive'], 'alpha': 5.0},
        pa_ilag_config={'lambda_attr': 0.0, 'sensitive_feature': 'sensitive'},
    )
    minimizer.fit(dataset=ArrayDataset(x, est.predict(x), features_names=feat))
    assert minimizer.security_report['pa_ilag']['num_divergence_steps'] > 0


# Full pipeline integration
def test_security_pipeline():
    x, labels = _make_dataset()
    feat = x.columns.tolist()
    est = DecisionTreeClassifier(random_state=0, min_samples_split=2, min_samples_leaf=1)
    est.fit(x, labels)
    minimizer = GeneralizeToRepresentative(
        estimator=est, target_accuracy=0.99, features_to_minimize=feat,
        weighted_ncp_config={'sensitive_features': ['sensitive'], 'alpha': 2.0},
        dp_config={'epsilon': 1.0, 'max_retries': 5, 'seed': 21},
        disclosure_config={'sensitive_feature': 'sensitive', 'auc_threshold': 0.8,
                           'test_size': 0.3, 'seed': 21},
        diversity_config={'sensitive_features': ['sensitive'], 'k_min': 10,
                          'l_min': 2, 't_threshold': 0.3},
        pa_ilag_config={'lambda_attr': 1.0, 'sensitive_feature': 'sensitive'},
    )
    minimizer.fit(dataset=ArrayDataset(x, est.predict(x), features_names=feat))
    rpt = minimizer.security_report

    # All four report sections must be present
    assert {'pa_ilag', 'diversity', 'randomized_representatives', 'disclosure'} <= set(rpt)
    assert {'feature_removal_order', 'baseline_removal_order', 'num_divergence_steps'} <= set(rpt['pa_ilag'])
    assert {'t_threshold', 'per_feature', 'per_cell_distances'} <= set(rpt['diversity'])

    # Default node-based transform path
    out = minimizer.transform(None, dataset=ArrayDataset(x, features_names=feat))
    assert out.shape == x.shape

    # Containment-mapping path activated after cell merges
    minimizer._use_cell_contains_mapping = True
    assert minimizer.transform(None, dataset=ArrayDataset(x, features_names=feat)).shape == x.shape
