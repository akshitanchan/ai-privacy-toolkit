[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/5836/badge)](https://bestpractices.coreinfrastructure.org/projects/5836)

Goldsteen et al. [1] introduced a method to minimise the personal data an ML model needs at inference time: generalise input features until accuracy just starts to drop, guided by an NCP-over-accuracy-gain score (ILAG). The method works well, but the paper flags security gaps it leaves open. This fork closes four of them, including a novel closed feedback loop (PA-ILAG).

## Features Introduced

### Feature 1: Weighted NCP

The original NCP treats every feature equally, so a sensitive column like *sex* can be the cheapest to un-generalise, and the first one exposed. `compute_sensitivity_weights()` in `apt/minimization/weighted_ncp.py` assigns sensitive features weight `1 + alpha`, rescaled to sum to `n`, and feeds these into the ILAG numerator so that stripping protection from sensitive columns costs more. This addresses the gap explored in Section 5.1.

### Feature 2: Leakage-Aware Feature Selection (PA-ILAG)

ILAG is blind to whether un-generalising a feature actually helps an attacker, since it optimises purely for accuracy versus information loss. PA-ILAG closes this gap by training a logistic-regression attacker [4] at each feature-removal step and feeding the resulting ROC-AUC into the score `PA-ILAG(f) = base_ilag(f) + λ · leakage_delta(f)`. Candidates whose un-generalisation would raise attacker AUC are penalised and deferred until safer features have been removed. The core functions are `compute_pa_ilag_score()`, `compute_sensitive_auc()`, and `measure_attribute_disclosure()` in `apt/minimization/security_metrics.py`.

The `security_report` tracks how this reordering diverges from plain ILAG (the `--ablation` flag automates the comparison). This appears to be the first integration of AUC-based inference risk into the ILAG criterion as a closed feedback loop. This addresses the gap explored in Sections 4.3, 5.3, and 5.4.

### Feature 3: Cell Privacy Stack

Decision-tree cells can end up too small or too homogeneous. `enforce_cell_privacy()` in `apt/minimization/security_postprocess.py` runs a two-layer defence in one call, targeting the gap identified in Section 3.1:

- **k-anonymity + l-diversity**: merges cells violating minimum record count or minimum distinct sensitive values [3] with same-label neighbours and rebuilds representatives.
- **t-closeness**: computes Earth Mover's Distance (numeric) or variational distance (categorical) between each cell's sensitive-attribute distribution and the global one [6]. This gives per-cell severity scores rather than a binary pass/fail.

### Feature 4: DP-Inspired Representative Randomization

Cell representatives in the original method are real training records, which creates a verbatim linkage risk. `randomize_cell_representatives()` in `apt/minimization/security_postprocess.py` replaces them with synthetic points using Laplace noise [2] for numerics and the exponential mechanism [5] for categoricals. Candidates are verified against training data to block exact matches. This follows the Laplace mechanism from Dwork et al. [2] but does not constitute formal (ε, δ)-DP, as noise is scaled per cell range rather than per-record sensitivity. This gap was documented in Section 2.1.

## Implementation

All four features integrate into the existing `fit()` pipeline in `apt/minimization/minimizer.py`. Additionally, unit tests (`tests/test_minimizer_security.py`) and a Jupyter notebook (`notebooks/minimization_security_extension.ipynb`) are included.

### How `fit()` ties it together

1. Build decision-tree cells (existing behaviour).
2. `compute_sensitivity_weights()` assigns per-feature weights (Feature 1).
3. `_get_feature_to_remove()` loop: weighted NCP + `compute_pa_ilag_score()` scoring (Features 1 + 2).
4. `enforce_cell_privacy()`: k/l merge, `attach_representatives_from_data()`, `measure_tcloseness()` (Feature 3).
5. `randomize_cell_representatives()`: Laplace + exponential mechanism noise (Feature 4).
6. `measure_attribute_disclosure()`: AUC before and after (Feature 2).

### Validity and Limitations

Each feature targets a distinct attack surface: weighted NCP and PA-ILAG counter attribute inference; l-diversity and t-closeness counter homogeneity attacks; DP randomization counters verbatim linkage. The logistic-regression attacker captures only linear decision boundaries, so AUC is a conservative lower bound on true inference risk [4]. Unresolved l-diversity violations are reported, never silently dropped.

### Reproducibility

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
4. Yeom, S., et al. Privacy risk in machine learning. *IEEE CSF*, 2018.
5. McSherry, F. and Talwar, K. Mechanism design via differential privacy. *FOCS*, 2007.
6. Li, N., Li, T., and Venkatasubramanian, S. t-closeness: Privacy beyond k-anonymity and l-diversity. *ICDE*, 2007.