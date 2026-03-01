[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/5836/badge)](https://bestpractices.coreinfrastructure.org/projects/5836)

# ai-privacy-toolkit
<p align="center">
  <img src="docs/images/logo with text.jpg?raw=true" width="467" title="ai-privacy-toolkit logo">
</p>
<br />

This repository extends IBM's `ai-privacy-toolkit` with four security features. Goldsteen et al. [1] optimize accuracy versus information loss (NCP) for GDPR data minimization. The paper identifies security gaps it does not address; each feature below closes one.

## Feature 1 - Sensitivity-Weighted NCP (Section 5.1 gap)

`apt/minimization/weighted_ncp.py` → `compute_sensitivity_weights()`. Sensitive features receive weight `1 + alpha`, rescaled to sum to `n`. These propagate into the ILAG numerator via `self._feature_weights`, making sensitive features costlier to un-generalize.

## Feature 2 - Privacy-Augmented Feature Selection (Section 4.3 + Section 5.3/5.4 gaps)

`apt/minimization/security_metrics.py` → `compute_pa_ilag_score()`, `compute_sensitive_auc()`, `measure_attribute_disclosure()`. A logistic-regression attacker [4, 5] reports ROC-AUC before and after generalization. The PA-ILAG formula `PA-ILAG(f) = base_ilag(f) + λ · leakage_delta(f)` penalizes features whose un-generalization raises attacker AUC. The `security_report` records `feature_removal_order` vs `baseline_removal_order` and `num_divergence_steps`. This is the primary novel contribution: to our knowledge, the first integration of AUC-based inference risk into the ILAG criterion as a closed feedback loop.

## Feature 3 - Cell Privacy Stack (Section 3.1 gap)

`apt/minimization/security_postprocess.py` → `enforce_cell_privacy()` orchestrates a three-level stack in a single call:

- **k-anonymity + l-diversity** (via `enforce_diversity` internally): merges cells violating `k_min` or `l_min` [3] with same-label neighbours, countering homogeneity attacks. Representatives are rebuilt after merges.
- **t-closeness** (via `measure_tcloseness` internally): quantifies distributional distance between each cell's sensitive-attribute distribution and the global distribution, per Li et al. [6]. Uses Earth Mover's Distance (numeric) and variational distance (categorical). Even when l-diversity merges report 0/510 changes, t-closeness provides per-cell severity scores (`max_distance`, `mean_distance`, violation counts), upgrading the stack from binary pass/fail to a graded risk profile.

## Feature 4 - DP-Inspired Representative Randomization (Section 2.1 gap)

`apt/minimization/security_postprocess.py` → `randomize_cell_representatives()` replaces verbatim training-record representatives with synthetic points using Laplace noise [2] (numeric) and exponential mechanism (categorical), verified against training data to block verbatim matches.

## `fit()` Call Sequence

1. Build decision-tree cells (existing behaviour).
2. `compute_sensitivity_weights` → `self._feature_weights` (Feature 1).
3. Feature-removal loop: weighted NCP + PA-ILAG scoring (Features 1+2).
4. `enforce_cell_privacy` → k/l merge + rep rebuild + t-closeness audit (Feature 3).
5. `randomize_cell_representatives` → DP replacement (Feature 4).
6. `measure_attribute_disclosure` → AUC before/after (Feature 2).

## Novelty Evidence

The `--ablation` mode runs four ILAG variants side-by-side: `raw_ilag`, `weighted_only`, `pa_only`, and `combined`. On German-credit (`target_accuracy=0.9`), `raw_ilag` un-generalizes the sensitive feature first because it has the lowest NCP cost. The `combined` variant overrides this: the PA-ILAG penalty detects that un-generalizing it raises attacker AUC and keeps it generalized longer. The differing `feature_removal_order` and non-zero `num_divergence_steps` confirm that the leakage-aware loop actively changes the generalization outcome, not just audits it.

## Security Validity and Limitations

The four features target distinct attack surfaces: weighted NCP and PA-ILAG counter attribute inference; l-diversity and t-closeness counter homogeneity attacks; DP randomization counters verbatim linkage. Logistic-regression AUC is a conservative lower bound on inference risk [5]. Representative randomization follows Dwork et al. [2] with per-cell ε but without record-level sensitivity analysis; this is DP-inspired, not formal (ε, δ)-DP. Unresolved l-diversity violations are counted and reported, not silently dropped.

## Reproducibility

```bash
pip install -r requirements.txt
pytest -q tests/test_minimizer_security.py
python scripts/run_security_extension.py --dataset german --target-accuracy 0.9
python scripts/run_security_extension.py --dataset german --target-accuracy 0.9 --ablation
```

## References

1. Goldsteen, A., et al. Data minimization for GDPR compliance in machine learning models. *AI and Ethics*, 2021.
2. Dwork, C., et al. Calibrating noise to sensitivity in private data analysis. *TCC*, 2006.
3. Machanavajjhala, A., et al. l-diversity: Privacy beyond k-anonymity. *TKDD*, 2007.
4. Fredrikson, M., et al. Model inversion attacks that exploit confidence information. *CCS*, 2015.
5. Yeom, S., et al. Privacy risk in machine learning. *IEEE CSF*, 2018.
6. Li, N., Li, T., and Venkatasubramanian, S. t-closeness: Privacy beyond k-anonymity and l-diversity. *ICDE*, 2007.