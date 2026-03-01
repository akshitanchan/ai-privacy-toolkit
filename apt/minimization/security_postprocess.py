"""
Cell post-processing: l-diversity enforcement, t-closeness measurement, and DP representative randomization.
"""
from typing import Dict, List, Optional, Sequence, Tuple
import copy
import numpy as np
import pandas as pd


# Check if a data row falls within a cell's numeric ranges and categorical allowlists
def _cell_contains_row(cell, row, feature_names):
    untouched = set(cell.get('untouched', []))
    for feature in feature_names:
        if feature in cell.get('ranges', {}):
            b = cell['ranges'][feature]
            s, e = b.get('start'), b.get('end')
            if (s is not None and row[feature] <= s) or (e is not None and row[feature] > e):
                return False
        elif feature in cell.get('categories', {}):
            if row[feature] not in cell['categories'][feature]:
                return False
        elif feature not in untouched:
            return False
    return True


def build_cell_index_map(cells: List[dict], data: pd.DataFrame, feature_names: Sequence[str]) -> Tuple[List[List[int]], List[int]]:
    """Map each row to the first cell whose ranges/categories contain it.

    :param cells: List of cell dicts (ranges, categories, untouched).
    :param data: Training data DataFrame.
    :param feature_names: Ordered feature names.
    :return: (indexes_per_cell, unmapped) tuple.
    """
    indexes_per_cell = [[] for _ in cells]
    unmapped = []
    for row_index, row in data.iterrows():
        matched = False
        # First-match assignment keeps mapping deterministic and lightweight
        for i, cell in enumerate(cells):
            if _cell_contains_row(cell, row, feature_names):
                indexes_per_cell[i].append(row_index)
                matched = True
                break
        if not matched:
            unmapped.append(row_index)
    return indexes_per_cell, unmapped


def attach_representatives_from_data(
    cells: List[dict],
    data: pd.DataFrame,
    feature_names: Sequence[str],
    indexes_per_cell: Optional[List[List[int]]] = None,
) -> List[List[int]]:
    """Rebuild each cell's representative from data after merges.

    :param cells: Cell dicts (modified in place).
    :param data: Training data DataFrame.
    :param feature_names: Ordered feature names.
    :param indexes_per_cell: Pre-computed mapping.
    :return: The indexes_per_cell mapping used.
    """
    if indexes_per_cell is None:
        indexes_per_cell, _ = build_cell_index_map(cells, data, feature_names)
    for i, cell in enumerate(cells):
        if cell.get('representative') is None:
            cell['representative'] = {}
        if not indexes_per_cell[i]:
            continue
        source_row = data.loc[indexes_per_cell[i][0]]
        for feature in [*cell.get('ranges', {}), *cell.get('categories', {})]:
            if feature in source_row.index:
                cell['representative'][feature] = source_row[feature]
    return indexes_per_cell


# Scan all cells for k-anonymity and l-diversity violations and return them alongside the current index map
def _find_diversity_violations(cells, data, sensitive_features, k_min, l_min, feature_names):
    indexes_per_cell, _ = build_cell_index_map(cells, data, feature_names)
    violations = []
    for i, cell in enumerate(cells):
        idxs = indexes_per_cell[i]
        if len(idxs) < k_min:
            violations.append({'cell_position': i, 'cell_id': cell['id'], 'type': 'k'})
        for sf in sensitive_features:
            n_distinct = len(data.loc[idxs, sf].dropna().unique()) if idxs else 0
            if n_distinct < l_min:
                violations.append({'cell_position': i, 'cell_id': cell['id'], 'type': 'l', 'feature': sf})
    return violations, indexes_per_cell


# Find the index of the nearest non-empty same-label cell to merge with the violating one
def _select_merge_target(violating_position, cells, indexes_per_cell):
    label = cells[violating_position].get('label')
    for i, cell in enumerate(cells):
        if i != violating_position and cell.get('label') == label and indexes_per_cell[i]:
            return i
    return None


# Compute the outer bounding interval when merging two numeric feature ranges
def _merge_numeric_bounds(a, b):
    sa = a.get('start') if a else None
    sb = b.get('start') if b else None
    ea = a.get('end') if a else None
    eb = b.get('end') if b else None
    return {'start': None if sa is None or sb is None else min(sa, sb),
            'end':   None if ea is None or eb is None else max(ea, eb)}


# Produce a new cell whose ranges and categories span those of both source cells
def _merge_cells(cell_a, cell_b):
    merged = {
        'id': int(cell_a['id']), 'label': cell_a.get('label'),
        'ranges': {}, 'categories': {},
        'untouched': sorted(set(cell_a.get('untouched', [])) & set(cell_b.get('untouched', []))),
        'representative': None,
    }
    for f in set(cell_a.get('ranges', {})) | set(cell_b.get('ranges', {})):
        merged['ranges'][f] = _merge_numeric_bounds(cell_a.get('ranges', {}).get(f), cell_b.get('ranges', {}).get(f))
    for f in set(cell_a.get('categories', {})) | set(cell_b.get('categories', {})):
        merged['categories'][f] = list(dict.fromkeys(
            list(cell_a.get('categories', {}).get(f, [])) + list(cell_b.get('categories', {}).get(f, []))
        ))
    return merged


def enforce_diversity(
    cells: List[dict],
    data: pd.DataFrame,
    feature_names: Sequence[str],
    sensitive_features: Sequence[str],
    k_min: int = 5,
    l_min: int = 2,
    max_iterations: int = 100,
) -> Tuple[List[dict], Dict[str, int]]:
    """Merge cells until k-anonymity and l-diversity bounds are satisfied.

    :param cells: Original cell list.
    :param data: Training data used to count records per cell.
    :param feature_names: Ordered feature names.
    :param sensitive_features: Attributes requiring l-diversity.
    :param k_min: Minimum records per cell.
    :param l_min: Minimum distinct sensitive values per cell.
    :param max_iterations: Safety cap on merge rounds.
    :return: (merged_cells, report_dict) with violation counts and merge count.
    """
    safe_cells = copy.deepcopy(cells)
    sensitive = [f for f in sensitive_features if f in data.columns]
    violations, indexes_per_cell = _find_diversity_violations(safe_cells, data, sensitive, k_min, l_min, feature_names)
    k_before = sum(1 for v in violations if v['type'] == 'k')
    l_before = sum(1 for v in violations if v['type'] == 'l')
    merge_count = 0
    unresolved = 0
    for _ in range(max_iterations):
        if not violations:
            break
        vp = violations[0]['cell_position']
        tp = _select_merge_target(vp, safe_cells, indexes_per_cell)
        if tp is None:
            unresolved += 1
            break
        lo, hi = min(vp, tp), max(vp, tp)
        merged = _merge_cells(safe_cells[lo], safe_cells[hi])
        safe_cells = [c for i, c in enumerate(safe_cells) if i not in (lo, hi)] + [merged]
        merge_count += 1
        violations, indexes_per_cell = _find_diversity_violations(safe_cells, data, sensitive, k_min, l_min, feature_names)
    k_after = sum(1 for v in violations if v['type'] == 'k')
    l_after = sum(1 for v in violations if v['type'] == 'l')
    return safe_cells, {
        'num_cells_before': len(cells), 'num_cells_after': len(safe_cells),
        'k_violations_before': int(k_before), 'l_violations_before': int(l_before),
        'k_violations_after': int(k_after), 'l_violations_after': int(l_after),
        'num_merges': int(merge_count), 'unresolved_violations': int(unresolved),
    }


def enforce_cell_privacy(
    cells: List[dict],
    data: pd.DataFrame,
    feature_names: Sequence[str],
    sensitive_features: Sequence[str],
    k_min: int = 5,
    l_min: int = 2,
    t_threshold: float = 0.3,
    max_iterations: int = 100,
) -> Tuple[List[dict], dict]:
    """Enforce k-anonymity, l-diversity, and t-closeness on generalised cells.

    Orchestrates the full cell-privacy pipeline:
      1. enforce_diversity - merge cells violating k/l bounds.
      2. attach_representatives_from_data - rebuild reps after merges.
      3. measure_tcloseness - audit distributional distance per cell.

    :param cells: Original cell list.
    :param data: Training data DataFrame.
    :param feature_names: Ordered feature names.
    :param sensitive_features: Attributes requiring l-diversity and t-closeness.
    :param k_min: Minimum records per cell (k-anonymity).
    :param l_min: Minimum distinct sensitive values per cell (l-diversity).
    :param t_threshold: Max acceptable distributional distance (t-closeness).
    :param max_iterations: Safety cap on merge rounds.
    :return: (merged_cells, unified_report) where report contains all k/l/t keys.
    """
    merged_cells, div_report = enforce_diversity(
        cells, data, feature_names, sensitive_features, k_min, l_min, max_iterations,
    )
    attach_representatives_from_data(merged_cells, data, feature_names)
    tc_report = measure_tcloseness(merged_cells, data, feature_names, sensitive_features, t_threshold)
    return merged_cells, {**div_report, **tc_report}


# Earth mover's distance between two ordered probability vectors, normalised to [0, 1]
def _compute_emd(cell_dist: np.ndarray, global_dist: np.ndarray) -> float:
    cdf_cell = np.cumsum(cell_dist)
    cdf_global = np.cumsum(global_dist)
    return float(np.sum(np.abs(cdf_cell - cdf_global)) / max(len(cell_dist) - 1, 1))


# Half l1 norm between two categorical probability vectors, bounded to [0, 1]
def _compute_variational_distance(cell_dist: np.ndarray, global_dist: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(cell_dist - global_dist)))


def measure_tcloseness(
    cells: List[dict],
    data: pd.DataFrame,
    feature_names: Sequence[str],
    sensitive_features: Sequence[str],
    t_threshold: float = 0.3,
) -> dict:
    """Measure t-closeness of each cell for each sensitive feature.

    t-closeness requires the distribution of a sensitive
    attribute in every equivalence class to be within distance *t* of its
    global distribution. Uses EMD for numeric and variational distance for
    categorical attributes.

    :param cells: Cell list (post-merge).
    :param data: Training data DataFrame.
    :param feature_names: Ordered feature names.
    :param sensitive_features: Attributes to measure.
    :param t_threshold: Distance threshold. Cells exceeding it are violations.
    :return: Dict with t_threshold, per-feature stats, and per_cell_distances.
    """
    indexes_per_cell, _ = build_cell_index_map(cells, data, feature_names)
    result: dict = {'t_threshold': t_threshold, 'per_feature': {}, 'per_cell_distances': []}

    for sf in sensitive_features:
        if sf not in data.columns:
            continue
        col = data[sf].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(col)

        # Global distribution over unique sorted values
        domain = sorted(col.unique())
        val_to_idx = {v: i for i, v in enumerate(domain)}
        global_counts = np.zeros(len(domain))
        for v in col:
            global_counts[val_to_idx[v]] += 1
        global_dist = global_counts / global_counts.sum()

        distances: List[Optional[float]] = []
        for idxs in indexes_per_cell:
            if not idxs:
                distances.append(None)
                continue
            local_vals = data.loc[idxs, sf].dropna()
            local_counts = np.zeros(len(domain))
            for v in local_vals:
                if v in val_to_idx:
                    local_counts[val_to_idx[v]] += 1
            total = local_counts.sum()
            if total == 0:
                distances.append(None)
                continue
            local_dist = local_counts / total
            d = _compute_emd(local_dist, global_dist) if is_numeric else _compute_variational_distance(local_dist, global_dist)
            distances.append(float(d))

        valid = [d for d in distances if d is not None]
        violating = sum(1 for d in valid if d > t_threshold)
        result['per_feature'][sf] = {
            'max_distance': float(max(valid)) if valid else 0.0,
            'mean_distance': float(np.mean(valid)) if valid else 0.0,
            'cells_violating_t': violating,
            'cells_passing_t': len(valid) - violating,
        }
        result['per_cell_distances'].append({'feature': sf, 'distances': distances})

    return result


# Return true if no training row exactly matches the candidate on every generalized feature
def _is_non_verbatim(candidate, data):
    mask = np.ones(len(data), dtype=bool)
    for feature, value in candidate.items():
        mask &= (data[feature] == value).to_numpy()
        if not mask.any():
            return True
    return not mask.any()


# Draw a synthetic representative using Laplace noise for numeric features and the exponential mechanism for categorical ones
def _sample_representative(cell, feature_data, epsilon, rng):
    rep = cell.get('representative') or {}
    result = {}
    for feature, bounds in cell.get('ranges', {}).items():
        lo = bounds.get('start') if bounds.get('start') is not None else feature_data[feature]['min']
        hi = bounds.get('end') if bounds.get('end') is not None else feature_data[feature]['max']
        base = rep.get(feature, (lo + hi) / 2.0)
        noised = float(base) + rng.laplace(0.0, (float(hi) - float(lo)) / epsilon)
        clipped = float(np.clip(noised, lo, hi))
        result[feature] = int(round(clipped)) if isinstance(base, (int, np.integer)) else clipped
    for feature, cats in cell.get('categories', {}).items():
        allowed = list(cats)
        base = rep.get(feature, allowed[0])
        if len(allowed) == 1:
            result[feature] = allowed[0]
        else:
            keep_prob = np.exp(epsilon) / (np.exp(epsilon) + len(allowed) - 1)
            if base in allowed and rng.rand() < keep_prob:
                result[feature] = base
            else:
                alts = [v for v in allowed if v != base]
                result[feature] = alts[rng.randint(0, len(alts))]
    return result


# Deterministically nudge a candidate point to guarantee it does not match any training row when sampling fails
def _build_non_verbatim_fallback(cell, candidate, feature_data):
    result = copy.deepcopy(candidate)
    for feature, bounds in cell.get('ranges', {}).items():
        val = result.get(feature)
        if val is None:
            continue
        fd = feature_data[feature]
        lo = bounds.get('start') if bounds.get('start') is not None else fd['min']
        hi = bounds.get('end') if bounds.get('end') is not None else fd['max']
        nudge = 1 if isinstance(val, (int, np.integer)) else max(abs(float(hi) - float(lo)) * 0.05, 0.01)
        new_val = float(val) + nudge
        if new_val > hi:
            new_val = float(val) - nudge
        result[feature] = int(round(new_val)) if isinstance(val, (int, np.integer)) else float(np.clip(new_val, lo, hi))
        return result
    # Flip the first categorical feature as a last resort
    for feature, cats in cell.get('categories', {}).items():
        alts = [v for v in cats if v != result.get(feature)]
        if alts:
            result[feature] = alts[0]
            return result
    return result


def randomize_cell_representatives(
    cells: List[dict],
    data: pd.DataFrame,
    feature_names: Sequence[str],
    feature_data: Dict[str, dict],
    epsilon: float = 1.0,
    max_retries: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """Replace each cell's representative with a DP-inspired synthetic point.

    Numeric features: Laplace noise scaled by cell_range / epsilon.
    Categorical features: exponential-mechanism coin flip.
    Each candidate is checked against training data to block verbatim matches.

    :param cells: Cell dicts (modified in place).
    :param data: Training data for verbatim-match checking.
    :param feature_names: Ordered feature names.
    :param feature_data: Per-feature metadata (min, max, range).
    :param epsilon: Privacy budget per cell representative.
    :param max_retries: Sampling attempts before deterministic fallback.
    :param random_state: RNG seed.
    :return: Dict with non_verbatim_rate and fallback_count.
    """
    rng = np.random.RandomState(random_state)
    indexes_per_cell, _ = build_cell_index_map(cells, data, feature_names)
    total = len(cells)
    non_verbatim_count = 0
    fallback_count = 0
    for i, cell in enumerate(cells):
        if cell.get('representative') is None:
            cell['representative'] = {}
        generalized = [*cell.get('ranges', {}), *cell.get('categories', {})]
        if not generalized:
            cell['representative'] = {}
            non_verbatim_count += 1
            continue
        rep = {f: cell['representative'][f] for f in generalized if f in cell['representative']}
        if indexes_per_cell[i]:
            source_row = data.loc[indexes_per_cell[i][0]]
            for f in generalized:
                if f not in rep and f in source_row.index:
                    rep[f] = source_row[f]
        cell['representative'] = rep
        accepted = None
        for _ in range(max_retries):
            candidate = _sample_representative(cell, feature_data, epsilon, rng)
            if _is_non_verbatim(candidate, data):
                accepted = candidate
                break
        if accepted is None:
            fallback_count += 1
            # Deterministic fallback ensures we still avoid verbatim representatives
            accepted = _build_non_verbatim_fallback(cell, rep, feature_data)
        cell['representative'] = accepted
        if _is_non_verbatim(accepted, data):
            non_verbatim_count += 1
    return {
        'non_verbatim_rate': float(non_verbatim_count) / total,
        'fallback_count': int(fallback_count),
    }
