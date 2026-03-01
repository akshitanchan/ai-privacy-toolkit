#!/usr/bin/env python
"""Run the minimization security extension on Adult or German datasets."""
import argparse

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from apt.minimization import GeneralizeToRepresentative
from apt.utils.dataset_utils import get_adult_dataset_pd, get_german_credit_dataset_pd
from apt.utils.datasets import ArrayDataset


# Load train/test splits and the sensitive feature name for the named dataset
def _load_dataset(name):
    if name == 'adult':
        (x_tr, y_tr), (x_te, y_te) = get_adult_dataset_pd()
        return x_tr, y_tr, x_te, y_te, 'sex'
    if name == 'german':
        (x_tr, y_tr), (x_te, y_te) = get_german_credit_dataset_pd()
        return x_tr, y_tr, x_te, y_te, 'Personal_status_sex'
    raise ValueError('Unsupported dataset: %s' % name)


# Build a column transformer that one-hot encodes categoricals and passes numeric columns through unchanged
def _create_encoder(features):
    num_cols = features.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [c for c in features.columns if c not in num_cols]
    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])
    return preprocessor, cat_cols

# Fit and evaluate one alpha/lambda security configuration, returning accuracy, the fitted minimizer, and its report
def _run_secure_variant(est, encoder, cat_feats, feat_names, x_train, x_test, train_preds, test_preds, target_accuracy, sensitive_feature, alpha, lambda_attr, security_verbose):
    minimizer = GeneralizeToRepresentative(
        estimator=est, target_accuracy=target_accuracy,
        categorical_features=cat_feats, encoder=encoder,
        features_to_minimize=feat_names,
        weighted_ncp_config={'sensitive_features': [sensitive_feature], 'alpha': alpha},
        dp_config={'epsilon': 1.0, 'max_retries': 10, 'seed': 42},
        disclosure_config={'sensitive_feature': sensitive_feature, 'auc_threshold': 0.7, 'seed': 42},
        diversity_config={'sensitive_features': [sensitive_feature], 'k_min': 5, 'l_min': 2, 't_threshold': 0.3},
        pa_ilag_config={'lambda_attr': lambda_attr, 'sensitive_feature': sensitive_feature},
        security_verbose=security_verbose,
    )
    minimizer.fit(dataset=ArrayDataset(x_train, train_preds, features_names=feat_names))
    sec_out = minimizer.transform(None, dataset=ArrayDataset(x_test, features_names=feat_names))
    sec_acc = float(est.score(encoder.transform(sec_out), test_preds))
    return sec_acc, minimizer, (minimizer.security_report or {})


# Run four fixed alpha/lambda variants side by side and print the comparison as novelty evidence
def _print_ablation_table(est, encoder, cat_feats, feat_names, x_train, x_test, train_preds, test_preds, target_accuracy, sensitive_feature):
    variants = [
        ('raw_ilag', 0.0, 0.0),
        ('weighted_only', 2.0, 0.0),
        ('pa_only', 0.0, 1.0),
        ('combined', 2.0, 1.0),
    ]

    print('\n6) Ablation (novelty evidence)')
    print('   %-13s %-33s %-10s %-10s %-11s %-8s' % (
        'variant', 'first3_order', 'div_steps', 'auc_after', 'rel_acc', 'ncp_fit',
    ))
    for name, alpha, lambda_attr in variants:
        sec_acc, minimizer, report = _run_secure_variant(
            est, encoder, cat_feats, feat_names, x_train, x_test, train_preds, test_preds,
            target_accuracy, sensitive_feature, alpha, lambda_attr, False
        )
        pa = report.get('pa_ilag') or {}
        disc = report.get('disclosure') or {}
        order_prefix = (pa.get('feature_removal_order') or [])[:3]
        auc_after = disc.get('auc_after')
        auc_text = 'n/a' if auc_after is None else '%.4f' % float(auc_after)
        ncp_fit = minimizer.ncp.fit_score
        ncp_text = 'n/a' if ncp_fit is None else '%.4f' % float(ncp_fit)
        print('   %-13s %-33s %-10s %-10s %-11.4f %-8s' % (
            name,
            str(order_prefix),
            pa.get('num_divergence_steps', 0),
            auc_text,
            sec_acc,
            ncp_text,
        ))


def run(dataset_name, target_accuracy, ablation=False):
    """Compare plain and security-extended minimization on a real dataset.

    Trains a decision tree on the chosen dataset, runs plain minimization as a
    reference, then re-runs with all four security features active. Prints a
    structured five-section report to the terminal covering accuracy, attribute
    disclosure AUC, diversity enforcement, DP representative randomization, and
    PA-ILAG divergence. Optionally appends a six-section ablation table that shows
    how each variant combination changes the feature-removal order.

    :param dataset_name: 'adult' or 'german'.
    :param target_accuracy: Relative accuracy threshold passed to the minimizer.
    :param ablation: When True, also prints the four-variant comparison table.
    """
    x_train, y_train, x_test, y_test, sens = _load_dataset(dataset_name)
    encoder, cat_feats = _create_encoder(x_train)

    encoder.fit(x_train)
    est = DecisionTreeClassifier(random_state=0)
    est.fit(encoder.transform(x_train), y_train)

    train_preds = est.predict(encoder.transform(x_train))
    test_preds = est.predict(encoder.transform(x_test))
    feat_names = x_train.columns.tolist()

    # Baseline (no security features)
    base = GeneralizeToRepresentative(
        estimator=est, target_accuracy=target_accuracy,
        categorical_features=cat_feats, encoder=encoder,
        features_to_minimize=feat_names,
    )
    base.fit(dataset=ArrayDataset(x_train, train_preds, features_names=feat_names))
    base_out = base.transform(None, dataset=ArrayDataset(x_test, features_names=feat_names))
    base_acc = float(est.score(encoder.transform(base_out), test_preds))

    # With all security features enabled
    sec_acc, sec, rpt = _run_secure_variant(
        est, encoder, cat_feats, feat_names, x_train, x_test, train_preds, test_preds,
        target_accuracy, sens, 2.0, 1.0, True
    )
    disc = rpt.get('disclosure') or {}
    div = rpt.get('diversity') or {}
    dp = rpt.get('randomized_representatives') or {}
    pa = rpt.get('pa_ilag') or {}

    # Terminal output
    print('\n' + '=' * 64)
    print('  Security Extension Results  --  %s' % dataset_name)
    print('=' * 64)
    print('\n1) Accuracy preservation')
    print('   target: %.4f   baseline: %.4f   secure: %.4f' % (
        target_accuracy, base_acc, sec_acc))
    print('\n2) Attribute disclosure (lower AUC = better privacy)')
    print('   AUC before: %s   after: %s   delta: %s   pass: %s' % (
        disc.get('auc_before', 'n/a'), disc.get('auc_after', 'n/a'),
        disc.get('auc_delta', 'n/a'), disc.get('threshold_pass', 'n/a')))
    print('\n3) Diversity (homogeneity-attack mitigation)')
    print('   k-violations  before: %s  after: %s' % (
        div.get('k_violations_before', 'n/a'), div.get('k_violations_after', 'n/a')))
    print('   l-violations  before: %s  after: %s   merges: %s' % (
        div.get('l_violations_before', 'n/a'), div.get('l_violations_after', 'n/a'),
        div.get('num_merges', 'n/a')))
    for sf_name, sf_stats in div.get('per_feature', {}).items():
        print('   t-closeness [%s]  max_dist: %.4f  mean_dist: %.4f  violating: %s  passing: %s' % (
            sf_name, sf_stats.get('max_distance', 0), sf_stats.get('mean_distance', 0),
            sf_stats.get('cells_violating_t', 'n/a'), sf_stats.get('cells_passing_t', 'n/a')))
    print('\n4) DP representative randomization')
    print('   non-verbatim rate: %.2f   fallback count: %s' % (
        dp.get('non_verbatim_rate', 0.0), dp.get('fallback_count', 'n/a')))
    print('\n5) PA-ILAG divergence')
    print('   PA-ILAG order: %s' % pa.get('feature_removal_order', []))
    print('   plain order:   %s' % pa.get('baseline_removal_order', []))
    print('   divergent steps: %s' % pa.get('num_divergence_steps', 0))
    if ablation:
        _print_ablation_table(
            est, encoder, cat_feats, feat_names, x_train, x_test, train_preds, test_preds, target_accuracy, sens
        )
    print('=' * 64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run minimization security extension.')
    parser.add_argument('--dataset', choices=['adult', 'german'], default='adult')
    parser.add_argument('--target-accuracy', type=float, default=0.98)
    parser.add_argument('--ablation', action='store_true',
                        help='Run four fixed variants and print novelty-evidence table.')
    args = parser.parse_args()
    run(args.dataset, args.target_accuracy, ablation=args.ablation)
