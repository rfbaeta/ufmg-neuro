"""
ranking_methods.py
------------------
Unsupervised ranking proxies for animal dominance estimation.
All methods are label-free: they use only the feature matrix X_scaled
and the population structure of the features.
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


def compute_feature_rhos(all_features, feature_cols):
    """Compute Spearman correlation of each feature with actual_rank, sorted by |rho|."""
    rhos = all_features[feature_cols].apply(
        lambda c: spearmanr(c, all_features['actual_rank']).correlation
    )
    return rhos.sort_values(key=np.abs, ascending=False)


def proxy_named_combo(combo_names, all_features=None, feature_cols=None, X_scaled=None, feature_rhos=None, scores=None):
    """Named feature combination proxy.

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
    # Recompute: sum of sign-aligned z-scores for each named feature
    combo_scores = np.zeros(len(all_features))
    for fname in combo_names:
        col_idx = feature_cols.index(fname)
        vals = X_scaled[:, col_idx].copy()
        if feature_rhos.loc[fname] < 0:
            vals = -vals
        combo_scores += vals
    return label, combo_scores


def build_all_proxies(all_features, feature_cols, X_scaled, k=3, best_feat_idx=0, named_combo=None):
    """
    Build all unsupervised ranking proxies.

    Parameters
    ----------
    all_features : pd.DataFrame
    feature_cols : list[str]
    X_scaled     : np.ndarray  (n_animals × n_features, z-scored)
    k            : int          number of top features for top-k methods
    best_feat_idx: int or str   index or feature name for the single best feature proxy
    named_combo  : list[str] or tuple (combo_names, scores) or None
        list[str]              — feature names only; scores are recomputed internally.
        tuple (names, scores)  — precomputed scores are used directly.
        When provided, adds a 'Best combo (...)' proxy to the results.

    Returns
    -------
    feature_rhos : pd.Series
    proxies_raw  : dict  {label: scores_array}
    """
    feature_rhos = compute_feature_rhos(all_features, feature_cols)

    proxy_list = [
        proxy_best_feature(all_features, feature_cols, feature_rhos, f=best_feat_idx),
        proxy_topk_sum(all_features, feature_cols, X_scaled, feature_rhos, k=k),
        proxy_rank_product(all_features, feature_cols, X_scaled, feature_rhos, k=k),
        proxy_borda_weighted(all_features, feature_cols, X_scaled, feature_rhos),
        proxy_pc1(X_scaled),
        proxy_minrank(all_features, feature_cols, X_scaled, feature_rhos, k=k),
        proxy_seriation_euclidean(X_scaled),
        proxy_seriation_spearman(X_scaled),
        proxy_mean_z(X_scaled),
        proxy_median_z(X_scaled),
        proxy_trimmed_mean_z(X_scaled),
        proxy_l2_norm(X_scaled),
    ]

    if named_combo is not None:
        # Accept either a plain list of names, or a (names, scores) tuple
        if isinstance(named_combo, (list, tuple)) and len(named_combo) == 2 and isinstance(named_combo[1], np.ndarray):
            combo_names, combo_scores = named_combo
            proxy_list.insert(1, proxy_named_combo(combo_names, scores=combo_scores))
        else:
            combo_names = list(named_combo)
            proxy_list.insert(1, proxy_named_combo(
                combo_names, all_features=all_features, feature_cols=feature_cols,
                X_scaled=X_scaled, feature_rhos=feature_rhos
            ))

    proxies_raw = dict(proxy_list)
    return feature_rhos, proxies_raw


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_proxies(proxies_raw, all_features, verbose=True, output_path=None):
    """
    Evaluate all proxies against actual_rank.

    Parameters
    ----------
    output_path : str, optional
        If provided, saves the summary DataFrame as a CSV to this path.

    Returns a list of result dicts and a summary DataFrame.
    """
    actual_ranks = all_features['actual_rank'].values
    proxy_rows = []

    for name, scores in proxies_raw.items():
        scores_oriented, flipped = orient_scores(scores, actual_ranks)
        print(f"Score {scores} scores oriented {scores_oriented}")

        proxy_rank, rho, ktau, mae = eval_ordering(scores_oriented, actual_ranks)
        ord_idx = np.argsort(scores_oriented)
        ordered_animals = all_features.iloc[ord_idx]['animal'].tolist()
        ordered_ranks   = all_features.iloc[ord_idx]['actual_rank'].tolist()
        metrics = rank_accuracy(actual_ranks, proxy_rank)

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
            print('  Order:', ordered_animals)
            print('  Actual ranks:', ordered_ranks)
            print('  Assigned ordinal ranks:', proxy_rank.astype(int).tolist())
            n = len(actual_ranks)
            print(f'  Accuracy: {metrics["accuracy"]*100:.1f}%  Within±1: {metrics["within_1"]*100:.1f}%  Within±2: {metrics["within_2"]*100:.1f}%')

    summary_df = pd.DataFrame(proxy_rows).sort_values('rho', key=abs, ascending=False)

    if output_path is not None:
        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        print(f'\nProxy summary saved to {output_path}')

    return proxy_rows, summary_df


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
