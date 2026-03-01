import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from apt.minimization import GeneralizeToRepresentative
from apt.minimization.security_metrics import compute_pa_ilag_score
from apt.minimization.security_postprocess import randomize_cell_representatives
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
