"""
ranking_methods.py
------------------
Ranking proxies for animal dominance estimation.

Some proxies are fully unsupervised and use only `X_scaled`, while others
use `feature_rhos` to select and sign-align features. Those `feature_rhos`
must come either from the current labeled dataset (`y`) or from a previously
saved labeled reference set when running inference on unlabeled data.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, kendalltau
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def eval_ordering(scores, y_true):
    """Evaluate an ordering proxy against ground-truth ranks.
    Returns pred_rank, rho, ktau, mae. Accuracy metrics are handled by rank_accuracy().
    """
    pred_rank = rankdata(scores, method='ordinal')
    rho = spearmanr(pred_rank, y_true).correlation
    ktau = kendalltau(pred_rank, y_true, variant='b').correlation
    mae = np.mean(np.abs(pred_rank - y_true))
    return pred_rank, rho, ktau, mae


import numpy as np
from scipy.stats import spearmanr

def rank_accuracy(actual_ranks, proxy_ranks):
    """Compare two integer rank arrays and return accuracy metrics.

    Parameters
    ----------
    actual_ranks : array-like of int
        Ground-truth rank for each animal (1 = most dominant).
    proxy_ranks  : array-like of int
        Predicted rank for each animal.

    Returns
    -------
    dict with keys:
        accuracy  – fraction of animals with exact rank match
        within_1  – fraction with |error| <= 1
        within_2  – fraction with |error| <= 2
        mae       – mean absolute error
        rho       – Spearman correlation
    """
    actual = np.asarray(actual_ranks, dtype=float)
    pred   = np.asarray(proxy_ranks,  dtype=float)
    abs_err = np.abs(pred - actual)
    return {
        'accuracy': float(np.mean(abs_err == 0)),
        'within_1': float(np.mean(abs_err <= 1)),
        'within_2': float(np.mean(abs_err <= 2)),
        'mae':      float(np.mean(abs_err)),
        'rho':      float(spearmanr(actual, pred).correlation),
    }


def order_by_rank(animal_names, actual_ranks, pred_ranks, by="animal"):
    """Reindex animals/actual/predicted ranks by numeric animal id or by ground-truth rank.

    Parameters
    ----------
    animal_names : list[str]
    actual_ranks  : array-like of int  (1 = most dominant)
    pred_ranks    : array-like of int
    by            : {"animal", "rank"}
        "animal" — order by numeric animal id (e.g. animal_2 before animal_11).
        "rank"   — order by ascending ground-truth rank (1 -> N).

    Returns
    -------
    animals_sorted      : list[str]  animals ordered per `by`
    ground_sorted       : np.ndarray  ground-truth ranks reindexed per `by`
    pred_sorted         : np.ndarray  predicted ranks reindexed per `by`
    animals_pred_sorted : list[str]  animals ordered by ascending predicted rank
    """
    actual_ranks = np.asarray(actual_ranks)
    pred_ranks = np.asarray(pred_ranks)

    if by == "animal":
        order = np.argsort([int(name.split("_")[1]) for name in animal_names])
    elif by == "rank":
        order = np.argsort(actual_ranks)
    else:
        raise ValueError(f"`by` must be 'animal' or 'rank', got {by!r}")

    animals_sorted = [animal_names[i] for i in order]
    pred_order = np.argsort(pred_ranks)
    animals_pred_sorted = [animal_names[i] for i in pred_order]
    return animals_sorted, actual_ranks[order], pred_ranks[order], animals_pred_sorted


# ---------------------------------------------------------------------------
# Individual proxy builders
# ---------------------------------------------------------------------------

def proxy_best_feature(all_features, feature_cols, feature_rhos, f=0):
    """Single best-correlated feature, sign-aligned so higher = more dominant.

    Parameters
    ----------
    f : int or str
        If int, selects the f-th feature in feature_rhos (ranked by |rho|).
        If str, selects the feature by name directly.
    """
    if isinstance(f, str):
        if f not in feature_rhos.index:
            raise ValueError(f"Feature '{f}' not found in feature_rhos. Available: {list(feature_rhos.index[:10])}...")
        best_feat = f
        rho_val = feature_rhos.loc[f]
    else:
        best_feat = feature_rhos.index[f]
        rho_val = feature_rhos.iloc[f]
    vals = all_features[best_feat].values.copy()
    if rho_val < 0:
        vals = -vals
    label = f'Best feature ({best_feat})'
    return label, vals


def proxy_combinek_sum(all_features, feature_cols, X_scaled, feature_rhos, indices=[0]):
    """Sum of sign-aligned z-scores for a custom selection of features.

    Parameters
    ----------
    indices : list of int
        Positional indices into feature_rhos (sorted by |rho|).
        e.g. [0, 3, 4] picks the 1st, 4th and 5th most correlated features.
    """
    selected_feats = [feature_rhos.index[i] for i in indices]
    scores = np.zeros(len(all_features))
    for fname in selected_feats:
        col_idx = feature_cols.index(fname)
        vals = X_scaled[:, col_idx].copy()
        if feature_rhos.loc[fname] < 0:
            vals = -vals
        scores += vals
    label = f'Custom sum indices={indices} ({", ".join(selected_feats)})'
    return label, scores


def proxy_topk_sum(all_features, feature_cols, X_scaled, feature_rhos, k=3):
    """Sum of top-k sign-aligned z-scores."""
    topk_feats = feature_rhos.index[:k]
    scores = np.zeros(len(all_features))
    for fname in topk_feats:
        col_idx = feature_cols.index(fname)
        vals = X_scaled[:, col_idx].copy()
        if feature_rhos.loc[fname] < 0:
            vals = -vals
        scores += vals
    label = f'Sum top{k} sign-aligned z'
    return label, scores


def proxy_rank_product(all_features, feature_cols, X_scaled, feature_rhos, k=3):
    """Geometric mean of ranks across top-k features (rank product)."""
    topk_feats = feature_rhos.index[:k]
    product = np.ones(len(all_features))
    for fname in topk_feats:
        col_idx = feature_cols.index(fname)
        vals = X_scaled[:, col_idx].copy()
        if feature_rhos.loc[fname] < 0:
            vals = -vals
        product *= rankdata(vals, method='average')
    scores = product ** (1.0 / k)
    label = f'Rank product (geom mean) top{k}'
    return label, scores


def proxy_borda_weighted(all_features, feature_cols, X_scaled, feature_rhos):
    """Borda rank aggregation weighted by |Spearman rho| across all features."""
    abs_rhos = np.abs(feature_rhos.values)
    abs_rhos_norm = abs_rhos / (abs_rhos.sum() + 1e-9)
    scores = np.zeros(len(all_features))
    for i, fname in enumerate(feature_rhos.index):
        col_idx = feature_cols.index(fname)
        vals = X_scaled[:, col_idx].copy()
        if feature_rhos.iloc[i] < 0:
            vals = -vals
        scores += abs_rhos_norm[i] * rankdata(vals, method='average')
    label = 'Borda weighted by |rho| (all feats)'
    return label, scores


def proxy_pc1(X_scaled):
    """Project onto PC1 of the full feature matrix (purely unsupervised)."""
    pca = PCA(n_components=1)
    scores = pca.fit_transform(X_scaled).ravel()
    label = 'PC1 projection (unsupervised)'
    return label, scores


def proxy_minrank(all_features, feature_cols, X_scaled, feature_rhos, k=3):
    """Min-rank across top-k features: conservative — animal must rank high in all."""
    topk_feats = feature_rhos.index[:k]
    scores = np.full(len(all_features), np.inf)
    for fname in topk_feats:
        col_idx = feature_cols.index(fname)
        vals = X_scaled[:, col_idx].copy()
        if feature_rhos.loc[fname] < 0:
            vals = -vals
        scores = np.minimum(scores, rankdata(vals, method='average'))
    label = f'Min-rank top{k} (conservative)'
    return label, scores


def proxy_seriation_euclidean(X_scaled):
    """Seriation via hierarchical clustering with euclidean distance + optimal leaf ordering."""
    dist = pdist(X_scaled, metric='euclidean')
    Z = linkage(dist, method='average')
    Z_opt = optimal_leaf_ordering(Z, dist)
    order = leaves_list(Z_opt)
    scores = rankdata(order, method='ordinal')
    label = 'Seriation euclidean (opt leaf)'
    return label, scores


def proxy_seriation_spearman(X_scaled):
    """Seriation via hierarchical clustering with Spearman distance + optimal leaf ordering."""
    spear_corr, _ = spearmanr(X_scaled, axis=1)
    spear_dist = 1 - spear_corr
    np.fill_diagonal(spear_dist, 0)
    cond = squareform(spear_dist, checks=False)
    Z = linkage(cond, method='average')
    Z_opt = optimal_leaf_ordering(Z, cond)
    order = leaves_list(Z_opt)
    scores = rankdata(order, method='ordinal')
    label = 'Seriation spearman (opt leaf)'
    return label, scores


def proxy_mean_z(X_scaled):
    """Arithmetic mean of z-scores across all features."""
    return 'Mean z across features', X_scaled.mean(axis=1)


def proxy_median_z(X_scaled):
    """Median of z-scores across all features."""
    return 'Median z across features', np.median(X_scaled, axis=1)


def proxy_trimmed_mean_z(X_scaled, lo_pct=10, hi_pct=90):
    """Trimmed mean of z-scores per animal (drop lo_pct/hi_pct extremes)."""
    q_low = np.percentile(X_scaled, lo_pct, axis=1)
    q_high = np.percentile(X_scaled, hi_pct, axis=1)
    scores = []
    for row, lo, hi in zip(X_scaled, q_low, q_high):
        mask = (row >= lo) & (row <= hi)
        scores.append(row[mask].mean() if mask.any() else row.mean())
    return f'Trimmed mean z ({lo_pct}-{hi_pct}%)', np.array(scores)


def proxy_l2_norm(X_scaled):
    """L2 norm of z-score vector per animal."""
    return 'L2 norm of z', np.linalg.norm(X_scaled, axis=1)


# ---------------------------------------------------------------------------
# High-level builder: compute feature_rhos + all proxies
# ---------------------------------------------------------------------------

def orient_scores(scores, actual_ranks):
    """Orient scores so that higher score → less dominant (higher rank number).
    This is the direction required by eval_ordering / rankdata so rank 1 = most dominant.

    Parameters
    ----------
    scores       : array-like  raw proxy scores for each animal
    actual_ranks : array-like  ground-truth ranks (1 = most dominant)

    Returns
    -------
    scores_oriented : np.ndarray  (flipped if necessary)
    direction_flipped : bool
    """
    scores = np.asarray(scores, dtype=float)
    rho_dir = spearmanr(scores, actual_ranks).correlation
    flipped = rho_dir is not None and rho_dir < 0
    return (-scores if flipped else scores), flipped


def compute_feature_rhos(all_features, feature_cols, y):
    """Compute Spearman correlation of each feature with actual_rank, sorted by |rho|."""
    rhos = all_features[feature_cols].apply(
        lambda c: spearmanr(c, y).correlation
    )
    return rhos.sort_values(key=np.abs, ascending=False)


def prepare_feature_rhos(feature_rhos, feature_cols):
    """Validate and normalize externally provided feature_rhos for inference use."""
    if not isinstance(feature_rhos, pd.Series):
        feature_rhos = pd.Series(feature_rhos)

    missing_features = [feature for feature in feature_cols if feature not in feature_rhos.index]
    if missing_features:
        raise ValueError(
            "Missing feature_rhos entries for: "
            + ", ".join(missing_features)
        )

    return feature_rhos.loc[feature_cols].sort_values(key=np.abs, ascending=False)


def _get_sign_aligned_combo_matrix(combo_names, feature_cols, X_scaled, feature_rhos):
    """Return sign-aligned z-score columns for the selected combo features."""
    combo_columns = []
    for fname in combo_names:
        col_idx = feature_cols.index(fname)
        vals = X_scaled[:, col_idx].copy()
        if feature_rhos.loc[fname] < 0:
            vals = -vals
        combo_columns.append(vals)
    return np.column_stack(combo_columns)


def proxy_named_combo(combo_names, all_features=None, feature_cols=None, X_scaled=None, feature_rhos=None, scores=None):
    """Named feature combination proxy using the sign-aligned sum.

    Parameters
    ----------
    combo_names : list[str]
        Feature names in the combination — always used for the label.
    scores : np.ndarray, optional
        Precomputed composite scores. If provided, used directly.
        If None, scores are recomputed from all_features/feature_cols/X_scaled/feature_rhos.
    """
    label = f'Best combo ({" + ".join(combo_names)})'
    if scores is not None:
        return label, np.asarray(scores, dtype=float)
    combo_matrix = _get_sign_aligned_combo_matrix(combo_names, feature_cols, X_scaled, feature_rhos)
    return label, combo_matrix.sum(axis=1)


def proxy_named_combo_mean(combo_names, feature_cols, X_scaled, feature_rhos):
    """Named feature combination proxy using the sign-aligned mean."""
    label = f'Combo sign-aligned mean ({" + ".join(combo_names)})'
    combo_matrix = _get_sign_aligned_combo_matrix(combo_names, feature_cols, X_scaled, feature_rhos)
    return label, combo_matrix.mean(axis=1)


def build_all_proxies(all_features, feature_cols, X_scaled, y=None, k=3, best_feat_idx=0, named_combo=None, feature_rhos=None):
    """
    Build ranking proxies.

    Parameters
    ----------
    all_features : pd.DataFrame
    feature_cols : list[str]
    X_scaled     : np.ndarray  (n_animals × n_features, z-scored)
    y            : array-like or None
        Ground-truth ranks for the current dataset. If provided, `feature_rhos`
        are computed from this dataset.
    k            : int          number of top features for top-k methods
    best_feat_idx: int or str   index or feature name for the single best feature proxy
    named_combo  : list[str] or tuple (combo_names, scores) or None
        list[str]              — feature names only; scores are recomputed internally.
        tuple (names, scores)  — precomputed scores are used directly.
        When provided, adds a 'Best combo (...)' proxy to the results.
    feature_rhos : pd.Series or None
        Precomputed feature-to-rank correlations from a labeled reference set.
        Use this during inference when the current dataset has no labels.

    Returns
    -------
    feature_rhos : pd.Series
    proxies_raw  : dict  {label: scores_array}
    """
    if feature_rhos is None:
        if y is None:
            raise ValueError(
                "Provide either `y` to compute feature_rhos on this dataset or "
                "precomputed `feature_rhos` from a labeled reference set."
            )
        feature_rhos = compute_feature_rhos(all_features, feature_cols, y)
    else:
        print("Using feature_rhos provided externally (e.g. from a reference dataset).")
        feature_rhos = prepare_feature_rhos(feature_rhos, feature_cols)

    proxy_list = [
        proxy_best_feature(all_features, feature_cols, feature_rhos, f=best_feat_idx),
        #proxy_topk_sum(all_features, feature_cols, X_scaled, feature_rhos, k=k),
        #proxy_rank_product(all_features, feature_cols, X_scaled, feature_rhos, k=k),
        #proxy_borda_weighted(all_features, feature_cols, X_scaled, feature_rhos),
        #proxy_pc1(X_scaled),
        #proxy_minrank(all_features, feature_cols, X_scaled, feature_rhos, k=k),
        #proxy_seriation_euclidean(X_scaled),
        #proxy_seriation_spearman(X_scaled),
        #proxy_mean_z(X_scaled),
        #proxy_median_z(X_scaled),
        #proxy_trimmed_mean_z(X_scaled),
        #proxy_l2_norm(X_scaled),
    ]

    if named_combo is not None:
        # Accept either a plain list of names, or a (names, scores) tuple
        if isinstance(named_combo, (list, tuple)) and len(named_combo) == 2 and isinstance(named_combo[1], np.ndarray):
            combo_names, combo_scores = named_combo
            proxy_list.insert(1, proxy_named_combo(combo_names, scores=combo_scores))
            proxy_list.insert(2, proxy_named_combo_mean(combo_names, feature_cols, X_scaled, feature_rhos))
        else:
            combo_names = list(named_combo)
            proxy_list.insert(1, proxy_named_combo(
                combo_names, all_features=all_features, feature_cols=feature_cols,
                X_scaled=X_scaled, feature_rhos=feature_rhos
            ))
            proxy_list.insert(2, proxy_named_combo_mean(combo_names, feature_cols, X_scaled, feature_rhos))

    proxies_raw = dict(proxy_list)
    return feature_rhos, proxies_raw


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_proxies(proxies_raw, all_features, actual_ranks, verbose=True, output_path=None):
    """
    Evaluate all proxies against actual_rank.

    Parameters
    ----------
    output_path : str, optional
        If provided, saves the summary DataFrame as a CSV to this path.

    Returns a list of result dicts and a summary DataFrame.
    """
    actual_ranks = np.asarray(actual_ranks)
    if len(actual_ranks) != len(all_features):
        raise ValueError(
            f"Length mismatch: got {len(actual_ranks)} actual ranks for {len(all_features)} animals."
        )

    proxy_rows = []
    results = []
    for name, scores in proxies_raw.items():
        scores_oriented, flipped = orient_scores(scores, actual_ranks)

        proxy_rank, rho, ktau, mae = eval_ordering(scores_oriented, actual_ranks)
        metrics = rank_accuracy(actual_ranks, proxy_rank)
        results.append({
            'name': name, 'ground': actual_ranks, 'pred': proxy_rank})

        row_aligned_df = pd.DataFrame({
            'animal': all_features['animal'].tolist(),
            'true_rank': actual_ranks.astype(int),
            'oriented_score': scores_oriented.astype(float),
            'pred_rank': proxy_rank.astype(int),
        })
        comparison_df = row_aligned_df.copy()
        comparison_df['abs_error'] = (comparison_df['pred_rank'] - comparison_df['true_rank']).abs()
        predicted_order = [
            f"{animal}:{pred_rank}"
            for animal, pred_rank in zip(row_aligned_df['animal'], row_aligned_df['pred_rank'])
        ]
        score_rank_df = comparison_df[['animal', 'oriented_score', 'pred_rank', 'true_rank', 'abs_error']]

        proxy_rows.append({
            'proxy': name,
            'rho': rho,
            'kendall_tau': ktau,
            'mae': metrics['mae'],
            'accuracy': metrics['accuracy'],
            'within_1': metrics['within_1'],
            'within_2': metrics['within_2'],
            'max_abs_dev': float(np.abs(proxy_rank - actual_ranks).max()),
            'direction_flipped': flipped,
        })

        if verbose:
            print(f'\n{name}:')
            print('  Predicted ranks are generated with: rankdata(scores_oriented, method="ordinal")')
            print(f'  Metrics: rho={rho:.3f}  tau={ktau:.3f}  mae={metrics["mae"]:.2f}  acc={metrics["accuracy"]*100:.1f}%')
            print('  Animals in DataFrame row order:', row_aligned_df['animal'].tolist())
            print('  True ranks aligned to DataFrame rows:', row_aligned_df['true_rank'].astype(int).tolist())
            print('  Predicted order aligned to DataFrame rows:', predicted_order)
            print('  Predicted ranks aligned to DataFrame rows:', row_aligned_df['pred_rank'].astype(int).tolist())
            animal_rank_map = {
                row['animal']: {
                    'true_rank': int(row['true_rank']),
                    'pred_rank': int(row['pred_rank']),
                    'abs_error': int(row['abs_error']),
                }
                for _, row in comparison_df.iterrows()
            }
            print('  Animal -> ranks:', animal_rank_map)
            print(f"\n  {'Animal':<12} {'Score':>10} {'Pred':>6} {'True':>6} {'Error':>6}")
            print('  ' + '-' * 48)
            for _, row in score_rank_df.iterrows():
                err = int(row['abs_error'])
                marker = '✓' if err == 0 else ('~' if err <= 1 else 'x')
                print(f"  {row['animal']:<12} {row['oriented_score']:>10.4f} {int(row['pred_rank']):>6} {int(row['true_rank']):>6} {err:>6}  {marker}")

            print(f"\n  {'Animal':<12} {'True':>6} {'Pred':>6} {'Error':>6}")
            print('  ' + '-' * 34)
            for _, row in comparison_df.iterrows():
                err = int(row['abs_error'])
                marker = '✓' if err == 0 else ('~' if err <= 1 else 'x')
                print(f"  {row['animal']:<12} {int(row['true_rank']):>6} {int(row['pred_rank']):>6} {err:>6}  {marker}")
            n = len(actual_ranks)
            print(f'  Accuracy: {metrics["accuracy"]*100:.1f}%  Within±1: {metrics["within_1"]*100:.1f}%  Within±2: {metrics["within_2"]*100:.1f}%')

    summary_df = pd.DataFrame(proxy_rows).sort_values('rho', key=abs, ascending=False)

    if output_path is not None:
        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        print(f'\nProxy summary saved to {output_path}')

    return proxy_rows, summary_df, results


def plot_proxy_summary(proxy_rows):
    """Bar chart of all proxy methods sorted by |Spearman rho|. Returns a Plotly figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = pd.DataFrame(proxy_rows).sort_values('rho', key=abs, ascending=True)

    colors_rho = ['#2E86AB' if r >= 0 else '#E84855' for r in df['rho']]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Spearman ρ with Actual Rank', 'Mean Absolute Error (MAE)'),
        horizontal_spacing=0.12
    )

    fig.add_trace(go.Bar(
        x=df['rho'], y=df['proxy'], orientation='h',
        marker_color=colors_rho, name='Spearman ρ'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['mae'], y=df['proxy'], orientation='h',
        marker_color='#F18F01', name='MAE'
    ), row=1, col=2)

    fig.update_layout(
        title='Unsupervised Proxy Methods — Performance Summary',
        height=max(400, 40 * len(df)),
        width=1200,
        showlegend=False
    )
    fig.update_xaxes(title_text='Spearman ρ', row=1, col=1)
    fig.update_xaxes(title_text='MAE (ranks)', row=1, col=2)

    return fig
