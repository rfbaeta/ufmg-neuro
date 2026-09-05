## IMPORT LIBRARIES ----------------------------------------------------------------------------------------------------
#https://github.com/nnc-ufmg/circadipy/blob/main/src/circadipy/analysis_examples/intellicage/intellicage_analysis.ipynb
import sys                                                                                                              # Import sys to add paths to libraries                                                                                                           # Import re to work with regular expressions
import glob                                                                                                             # Import glob to read files                                                                                                   # Import numpy to work with arrays and make calculations                                                                                            # Import time to measure time
import os                                                                                                               # Import path to work with paths                
import pandas as pd
#from ranking_methods import build_all_proxies, evaluate_proxies
import numpy as np
from scipy import stats
from ranking_methods import build_all_proxies, evaluate_proxies
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from models import _select_top_k_features
                                                                               # Import pandas to work with dataframes
import warnings                                                                                                         # Import warnings to ignore warnings
warnings.filterwarnings('ignore')                                                                                       # Ignore warnings

## IMPORT CIRCADIPY ----------------------------------------------------------------------------------------------------

parent_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(parent_path)

## PCA Visualization - Dimensionality Reduction
from models import train_rf, train_gb, train_ridge, train_adaboost, train_extratrees, train_logistic
from summary import generate_summary_report
from util import  get_data_scaled, generate_features, generate_temporal_features, calculate_correlations, build_animal_protocols, get_sorted_animals_files, combine_all_features
from analysis_visualization import generate_methods_comparison, plot_animals_activity, plot_correlation, plot_pca_analysis, plot_cross_correlation, summary_visualization
from config import zt_0_time, labels_dict, beat_feat_idx_dict, beat_indices_dict

ranks_male = {
    "animal_5": 5,
    "animal_12": 8,
    "animal_9": 6,
    "animal_13": 4,
    "animal_15": 1,
    "animal_3": 3,
    "animal_8": 7,
    "animal_11": 2
}

ranks_female = {
    "animal_21": 1,
    "animal_18": 8,
    "animal_19": 9,
    "animal_24": 6,
    "animal_22": 4,
    "animal_26": 7,
    "animal_23": 2,
    "animal_17": 5,
    "animal_16": 3
}

#usando su

def generate_model_features(animals_protocols, ranks, gender, output_folder):
    

    features_df = generate_features(
        animals_protocols, 
        ranks, 
        output_path=f'{output_folder}/basic_features_{gender}.csv'
    )

    temporal_df = generate_temporal_features(
        animals_protocols, 
        ranks, 
        output_path=f'{output_folder}/temporal_features_{gender}.csv'
    )

    all_features, feature_cols = combine_all_features(
        features_df, 
        temporal_df, 
        output_path=f'{output_folder}/all_features_{gender}.csv'
    )

    return all_features, feature_cols


def build_protocol(data_folder):

    individual_files = glob.glob(data_folder + "/unwrapped_data/**/*.txt", recursive=True)
    animals = [int(k.split("_")[1]) for k in ranks.keys()]
    animals_files = get_sorted_animals_files(individual_files, animals)
    animals_protocols, animals_by_day = build_animal_protocols(animals_files, zt_0_time=zt_0_time, labels_dict=labels_dict)
    return animals_protocols, animals_by_day


def plot_analysis(
        all_features, 
        feature_cols, 
        animals_protocols, 
        ranks, 
        corr_df,
        output_folder):
    
    output_corr = f"{output_folder}/feature_correlations_{gender}.html"
    fig_corr = plot_correlation(corr_df, output_corr, show=False)

    male_comparison_output = f"{output_folder}/animal_comparison_all_{gender}.html"
    fig_activity = plot_animals_activity(
        animals_protocols, 
        ranks, 
        output_path=male_comparison_output,
        show=False
    )

    output_path = f"{output_folder}/pca_visualization_{gender}.html"
    fig_pca = plot_pca_analysis(
        all_features, 
        feature_cols, 
        output_path=output_path,
        show=False
    )

    output_path = f"{output_folder}/cross_correlation_matrix_{gender}.html"
    corr_sync, pval_sync, fig_cross = plot_cross_correlation(
        animals_protocols, 
        ranks, 
        output_path=output_path,
        show=False
    )

    return corr_sync, pval_sync


def permutation_spearman_pvalue(y_true, y_pred, n_perm=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    observed_rho = stats.spearmanr(y_true, y_pred)[0]
    perm_rhos = []
    for _ in range(n_perm):
        perm = rng.permutation(y_true)
        perm_rhos.append(stats.spearmanr(perm, y_pred)[0])
    perm_rhos = np.array(perm_rhos)
    p_value = (np.sum(np.abs(perm_rhos) >= abs(observed_rho)) + 1) / (n_perm + 1)
    return observed_rho, p_value


def bootstrap_mae_ci(y_true, y_pred, n_boot=2000, random_state=42, alpha=0.05):
    rng = np.random.default_rng(random_state)
    maes = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        maes.append(np.mean(np.abs(y_true[idx] - y_pred[idx])))
    ci_low, ci_high = np.percentile(maes, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(np.mean(np.abs(y_true - y_pred))), float(ci_low), float(ci_high)


def train_models(all_features, feature_cols, X_scaled, gender, output_folder, p=3):

    model_folder = f"{output_folder}/models"
    os.makedirs(model_folder, exist_ok=True)

    output_path = f"{model_folder}/feature_importance_gb.html"
    gb, gb_mae, all_features, fig_gb = train_gb(all_features, feature_cols, X_scaled, output_path=output_path, p=p, show_plot=False)

    output_path = f"{model_folder}/feature_importance_ridge.html"
    ridge, ridge_mae, all_features, fig_ridge = train_ridge(all_features, feature_cols, X_scaled, output_path=output_path, p=p, show_plot=False )

    output_path = f"{model_folder}/feature_importance.html"
    rf, mae, all_features, fig_rf = train_rf(all_features, feature_cols, X_scaled, output_path=output_path, p=p, show_plot=False)

    output_path = f"{model_folder}/feature_importance_adaboost.html"
    ada_model, ada_mae, all_features, fig_ada = train_adaboost(all_features, feature_cols, X_scaled, output_path=output_path, p=p, show_plot=False)

    output_path = f"{model_folder}/feature_importance_extratrees.html"
    et_model, et_mae, all_features, fig_et = train_extratrees(all_features, feature_cols, X_scaled, output_path=output_path, p=p, show_plot=False)

    output_path = f"{model_folder}/feature_importance_logreg.html"
    logreg_model, logreg_acc, all_features, fig_logreg = train_logistic(all_features, feature_cols, X_scaled, output_path=output_path, p=p, show_plot=False)

    K = 10  # must match k used in the train_* calls

    y_true = all_features['actual_rank'].values
    models = {
        'rf_rank': train_rf,
        'gb_rank': train_gb,
        'ridge_rank': train_ridge,
        'ada_rank': train_adaboost,
        'et_rank': train_extratrees, 
        'logreg_rank': train_logistic,
    }

    # Pre-select the top-K features once (same selection used inside every trainer)
    _, X_topk = _select_top_k_features(all_features, feature_cols, X_scaled, K)

    cv = LeaveOneOut()
    results = []
    for name, trainer in models.items():
        model, _, _, _ = trainer(all_features.copy(), feature_cols, X_scaled, show_plot=False, k=K)
        y_pred_cv = cross_val_predict(model, X_topk, y_true, cv=cv, method='predict')
        rho, pval = permutation_spearman_pvalue(y_true, y_pred_cv, n_perm=2000)
        mae, ci_low, ci_high = bootstrap_mae_ci(y_true, y_pred_cv, n_boot=2000)
        results.append((name, rho, pval, mae, ci_low, ci_high))



    # Save LOO-CV summary to CSV
    loo_summary_df = pd.DataFrame(results, columns=['model', 'spearman_rho', 'pval', 'mae', 'ci_low', 'ci_high'])
    loo_summary_path = f"{model_folder}/loo_cv_summary_{gender}.csv"
    loo_summary_df.to_csv(loo_summary_path, index=False)
    print(f"Saved LOO-CV summary to {loo_summary_path}")
    print(loo_summary_df.to_string(index=False))
   
    output_path = f'{model_folder}/accuracy_metrics.csv'
    fig_accuracy = generate_methods_comparison(all_features, model_folder, show=False)






def main(data_folder, ranks, gender, output_folder, k=3, best_feat_idx=0, indices=None, p=3):

    

    animals_protocols, animals_by_day = build_protocol(data_folder)
    all_features, feature_cols = generate_model_features(
        animals_protocols, 
        ranks, 
        gender, 
        output_folder
    )

    X_scaled =  get_data_scaled(all_features, feature_cols)
    y = all_features['actual_rank'].values

    corr_df = calculate_correlations(
        all_features, 
        feature_cols, 
        output_path=f'{output_folder}/correlations_{gender}.csv'
    )

    feature_rhos, proxies_raw = build_all_proxies(
        all_features, 
        feature_cols, X_scaled, k=k, 
        best_feat_idx=best_feat_idx, 
        indices=indices
    )


    proxy_rows, proxy_summary = evaluate_proxies(
        proxies_raw, 
        all_features, 
        y,
        verbose=True,
        output_path=f'{output_folder}/proxy_evaluation_{gender}.csv'
    )

    try:
        corr_sync, pval_sync = plot_analysis(
            all_features, 
            feature_cols, 
            animals_protocols, 
            ranks, 
            corr_df,
            output_folder
        )
    except Exception as e:
        print(f"Error during analysis visualization: {e}")
    

    train_models(all_features, feature_cols, X_scaled, gender, output_folder, p=p)

    pass


if __name__ == "__main__":

    gender = 'female'
    ranks = ranks_male if gender == 'male' else ranks_female
    data_folder = "./data"          

    output_folder = f"./test_results_2026/{gender}"      
    os.makedirs(output_folder, exist_ok=True)                 
    
    main(
        data_folder=data_folder, 
        ranks=ranks, 
        gender=gender, 
        output_folder=output_folder,
        best_feat_idx=beat_feat_idx_dict[gender],
        indices=beat_indices_dict[gender]
    )  
