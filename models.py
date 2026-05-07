## Random Forest - Feature Importance (Supervised)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LogisticRegression

from sklearn.model_selection import cross_val_score, LeaveOneOut, LeavePOut
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def _select_top_k_features(all_features, feature_cols, X_scaled, k):
    """Return (selected_feature_cols, X_topk) using the k features most correlated with actual_rank."""
    from scipy.stats import spearmanr
    rhos = {col: abs(spearmanr(all_features[col], all_features['actual_rank']).statistic) for col in feature_cols}
    top_cols = sorted(rhos, key=rhos.get, reverse=True)[:k]
    top_idx = [feature_cols.index(c) for c in top_cols]
    return top_cols, X_scaled[:, top_idx]


def train_gb(all_features, feature_cols, X_scaled, output_path=None, show_plot=True, k=10, p=2):
    # Train Gradient Boosting with conservative settings for small datasets
    y = all_features['actual_rank'].values

    feature_cols, X_scaled = _select_top_k_features(all_features, feature_cols, X_scaled, k)
    print(f"Using top-{k} features: {feature_cols}")

    gb = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_scaled, y)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': gb.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nGradient boosting Feature Importance:")
    print(feature_importance)

    # Visualize

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_importance['feature'][:10],
        y=feature_importance['importance'][:10]
    ))
    fig.update_layout(
        title='Top 10 Most Important Features for Rank Prediction',
        xaxis_title='Feature',
        yaxis_title='Importance',
        height=500,
        width=1200
    )
    if output_path is not None:
        fig.write_html(output_path)
        
    if show_plot:
        fig.show()


    # Cross-validation with Leave-P-Out
    cv = LeaveOneOut() if p == 1 else LeavePOut(p)
    cv_scores = cross_val_score(gb, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
    mae = -cv_scores.mean()

    print(f"\nLeave-{p}-Out Cross-Validation:")
    print(f"Mean Absolute Error: {mae:.2f} ranks")
    print(f"Std: {cv_scores.std():.2f}")

    # Predict ranks
    predicted_ranks = gb.predict(X_scaled)
    all_features['gb_rank'] = predicted_ranks
    all_features['gb_rank'] = all_features['gb_rank'].rank(method='dense').astype(int)

    print("\nActual vs Predicted Ranks:")
    print(all_features[['animal', 'actual_rank', 'gb_rank']].sort_values('actual_rank'))

    return gb, mae, all_features, fig
    

def train_rf(all_features, feature_cols, X_scaled, output_path=None, show_plot=True, k=10, p=2):
    # Train Random Forest to predict rank
    y = all_features['actual_rank'].values

    feature_cols, X_scaled = _select_top_k_features(all_features, feature_cols, X_scaled, k)
    print(f"Using top-{k} features: {feature_cols}")

    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=3)
    rf.fit(X_scaled, y)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nRandom Forest Feature Importance:")
    print(feature_importance)

    # Visualize

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_importance['feature'][:10],
        y=feature_importance['importance'][:10]
    ))
    fig.update_layout(
        title='Top 10 Most Important Features for Rank Prediction',
        xaxis_title='Feature',
        yaxis_title='Importance',
        height=500,
        width=1200
    )
    if output_path is not None:
        fig.write_html(output_path)
        
    if show_plot:
        fig.show()

    # Cross-validation with Leave-P-Out
    cv = LeaveOneOut() if p == 1 else LeavePOut(p)
    cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
    mae = -cv_scores.mean()

    print(f"\nLeave-{p}-Out Cross-Validation:")
    print(f"Mean Absolute Error: {mae:.2f} ranks")
    print(f"Std: {cv_scores.std():.2f}")

    # Predict ranks
    predicted_ranks = rf.predict(X_scaled)
    all_features['rf_rank'] = predicted_ranks
    all_features['rf_rank'] = all_features['rf_rank'].rank(method='dense').astype(int)

    print("\nActual vs Predicted Ranks:")
    print(all_features[['animal', 'actual_rank', 'rf_rank']].sort_values('actual_rank'))

    return rf, mae, all_features, fig

def train_ridge(all_features, feature_cols, X_scaled, output_path=None, show_plot=True, alpha=1.0, k=10, p=2):
    """Train a Ridge regressor as a low-variance baseline for tiny datasets."""

    y = all_features['actual_rank'].values

    feature_cols, X_scaled = _select_top_k_features(all_features, feature_cols, X_scaled, k)
    print(f"Using top-{k} features: {feature_cols}")

    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_scaled, y)


    # Coefficients as importance proxy
    coef_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(ridge.coef_)
    }).sort_values('importance', ascending=False)

    print("\nRidge Coefficient Magnitudes:")
    print(coef_importance)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=coef_importance['feature'][:10],
        y=coef_importance['importance'][:10]
    ))
    fig.update_layout(
        title='Top Features by Ridge Coefficient Magnitude',
        xaxis_title='Feature',
        yaxis_title='|Coefficient|',
        height=500,
        width=1200
    )
    if output_path is not None:
        fig.write_html(output_path)
    if show_plot:
        fig.show()

    # Cross-validation with Leave-P-Out
    cv = LeaveOneOut() if p == 1 else LeavePOut(p)
    cv_scores = cross_val_score(
        ridge,
        X_scaled,
        y,
        cv=cv,
        scoring='neg_mean_absolute_error'
    )
    mae = -cv_scores.mean()
    print(f"\nLeave-{p}-Out Cross-Validation (Ridge):")
    print(f"Mean Absolute Error: {mae:.2f} ranks")
    print(f"Std: {cv_scores.std():.2f}")

    # Predict ranks
    predicted_ranks = ridge.predict(X_scaled)
    all_features['ridge_rank'] = predicted_ranks
    all_features['ridge_rank'] = all_features['ridge_rank'].rank(method='dense').astype(int)

    print("\nActual vs Predicted Ranks (Ridge):")
    print(all_features[['animal', 'actual_rank', 'ridge_rank']].sort_values('actual_rank'))

    return ridge, mae, all_features, fig


def train_adaboost(all_features, feature_cols, X_scaled, output_path=None, show_plot=True, k=10, p=2):
    """AdaBoost regressor with shallow trees for tiny datasets."""

    y = all_features['actual_rank'].values

    feature_cols, X_scaled = _select_top_k_features(all_features, feature_cols, X_scaled, k)
    print(f"Using top-{k} features: {feature_cols}")

    ada = AdaBoostRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )
    ada.fit(X_scaled, y)

    # Feature importance (mean over estimators)
    importances = np.mean([est.feature_importances_ for est in ada.estimators_], axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nAdaBoost Feature Importance (mean over estimators):")
    print(feature_importance)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_importance['feature'][:10],
        y=feature_importance['importance'][:10]
    ))
    fig.update_layout(
        title='Top Features for Rank Prediction (AdaBoost)',
        xaxis_title='Feature',
        yaxis_title='Importance',
        height=500,
        width=1200
    )
    if output_path is not None:
        fig.write_html(output_path)
    if show_plot:
        fig.show()

    cv = LeaveOneOut() if p == 1 else LeavePOut(p)
    cv_scores = cross_val_score(ada, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
    mae = -cv_scores.mean()
    print(f"\nLeave-{p}-Out Cross-Validation (AdaBoost):")
    print(f"Mean Absolute Error: {mae:.2f} ranks")
    print(f"Std: {cv_scores.std():.2f}")

    predicted_ranks = ada.predict(X_scaled)
    all_features['ada_rank'] = predicted_ranks
    all_features['ada_rank'] = all_features['ada_rank'].rank(method='dense').astype(int)

    print("\nActual vs Predicted Ranks (AdaBoost):")
    print(all_features[['animal', 'actual_rank', 'ada_rank']].sort_values('actual_rank'))

    return ada, mae, all_features, fig


def train_extratrees(all_features, feature_cols, X_scaled, output_path=None, show_plot=True, k=10, p=2):
    """Extra Trees regressor with shallow trees for small datasets."""

    y = all_features['actual_rank'].values

    feature_cols, X_scaled = _select_top_k_features(all_features, feature_cols, X_scaled, k)
    print(f"Using top-{k} features: {feature_cols}")

    et = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42
    )
    et.fit(X_scaled, y)

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': et.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nExtra Trees Feature Importance:")
    print(feature_importance)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_importance['feature'][:10],
        y=feature_importance['importance'][:10]
    ))
    fig.update_layout(
        title='Top Features for Rank Prediction (Extra Trees)',
        xaxis_title='Feature',
        yaxis_title='Importance',
        height=500,
        width=1200
    )
    if output_path is not None:
        fig.write_html(output_path)
    if show_plot:
        fig.show()

    cv = LeaveOneOut() if p == 1 else LeavePOut(p)
    cv_scores = cross_val_score(et, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
    mae = -cv_scores.mean()
    print(f"\nLeave-{p}-Out Cross-Validation (Extra Trees):")
    print(f"Mean Absolute Error: {mae:.2f} ranks")
    print(f"Std: {cv_scores.std():.2f}")

    predicted_ranks = et.predict(X_scaled)
    all_features['et_rank'] = predicted_ranks
    all_features['et_rank'] = all_features['et_rank'].rank(method='dense').astype(int)

    print("\nActual vs Predicted Ranks (Extra Trees):")
    print(all_features[['animal', 'actual_rank', 'et_rank']].sort_values('actual_rank'))

    return et, mae, all_features, fig


def train_logistic(all_features, feature_cols, X_scaled, output_path=None, show_plot=True, penalty='l2', C=1.0, k=10, p=2):
    """Multinomial/OVR logistic regression as a simple classifier for rank labels."""

    y = all_features['actual_rank'].values.astype(int)

    feature_cols, X_scaled = _select_top_k_features(all_features, feature_cols, X_scaled, k)
    print(f"Using top-{k} features: {feature_cols}")

    logreg = LogisticRegression(
        multi_class='ovr',
        penalty=penalty,
        C=C,
        max_iter=1000,
        random_state=42
    )
    logreg.fit(X_scaled, y)

    # Coefficients per class (use absolute sum as a simple importance proxy)
    coef_importance = pd.Series(np.abs(logreg.coef_).sum(axis=0), index=range(X_scaled.shape[1]))
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': coef_importance.values
    }).sort_values('importance', ascending=False)

    print("\nLogistic Regression Feature (Coeff) Magnitudes:")
    print(feature_importance)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_importance['feature'][:10],
        y=feature_importance['importance'][:10]
    ))
    fig.update_layout(
        title='Top Features by Logistic Coefficient Magnitude',
        xaxis_title='Feature',
        yaxis_title='|Coefficient|',
        height=500,
        width=1200
    )
    if output_path is not None:
        fig.write_html(output_path)
    if show_plot:
        fig.show()

    # Leave-P-Out CV accuracy
    cv = LeaveOneOut() if p == 1 else LeavePOut(p)
    cv_scores = cross_val_score(
        logreg,
        X_scaled,
        y,
        cv=cv,
        scoring='accuracy'
    )
    acc = cv_scores.mean()
    print(f"\nLeave-{p}-Out Cross-Validation (Logistic):")
    print(f"Accuracy: {acc:.2%}")
    print(f"Std: {cv_scores.std():.3f}")

    # Predict classes as ranks
    predicted_ranks = logreg.predict(X_scaled)
    all_features['logreg_rank'] = predicted_ranks

    print("\nActual vs Predicted Ranks (Logistic):")
    print(all_features[['animal', 'actual_rank', 'logreg_rank']].sort_values('actual_rank'))

    return logreg, acc, all_features, fig
# def train_xgboost(all_features, feature_cols, X_scaled, output_path=None, show_plot=True):
#     """Train XGBoost regressor, report CV MAE, and add ranked predictions to all_features."""

#     y = all_features['actual_rank'].values

#     xgb = XGBRegressor(
#         n_estimators=300,
#         learning_rate=0.05,
#         max_depth=3,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42
#     )
#     xgb.fit(X_scaled, y)

#     # Feature importance
#     feature_importance = pd.DataFrame({
#         'feature': feature_cols,
#         'importance': xgb.feature_importances_
#     }).sort_values('importance', ascending=False)

#     print("\nXGBoost Feature Importance:")
#     print(feature_importance)

#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=feature_importance['feature'][:10],
#         y=feature_importance['importance'][:10]
#     ))
#     fig.update_layout(
#         title='Top 10 Most Important Features for Rank Prediction (XGBoost)',
#         xaxis_title='Feature',
#         yaxis_title='Importance',
#         height=500,
#         width=1200
#     )
#     if output_path is not None:
#         fig.write_html(output_path)
#     if show_plot:
#         fig.show()

#     # Cross-validation with Leave-One-Out
#     loo = LeaveOneOut()
#     cv_scores = cross_val_score(
#         xgb,
#         X_scaled,
#         y,
#         cv=loo,
#         scoring='neg_mean_absolute_error'  # returns negatives for loss
#     )
#     mae = -cv_scores.mean()
#     print(f"\nLeave-One-Out Cross-Validation:")
#     print(f"Mean Absolute Error: {mae:.2f} ranks")
#     print(f"Std: {cv_scores.std():.2f}")

#     # Predict ranks
#     predicted_ranks = xgb.predict(X_scaled)
#     all_features['xgb_rank'] = predicted_ranks
#     all_features['xgb_rank'] = all_features['xgb_rank'].rank(method='dense').astype(int)

#     print("\nActual vs Predicted Ranks:")
#     print(all_features[['animal', 'actual_rank', 'xgb_rank']].sort_values('actual_rank'))

#     return xgb, mae, all_features
