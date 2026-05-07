from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from scipy.signal import correlate
import numpy as np
import pandas as pd
from scipy import stats
from util import get_data_scaled
import os


def plot_pca_analysis(all_features, feature_cols, output_path=None, show=True):

    # Prepare data for PCA (exclude animal and rank columns)
    X_scaled =  get_data_scaled(all_features, feature_cols)

    # Perform PCA
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)

    # Create visualization
    fig = go.Figure()

    for i, animal in enumerate(all_features['animal']):
        rank = all_features.iloc[i]['actual_rank']
        fig.add_trace(go.Scatter(
            x=[X_pca[i, 0]],
            y=[X_pca[i, 1]],
            mode='markers+text',
            marker=dict(size=15, color=rank, colorscale='Viridis', showscale=True, 
                    colorbar=dict(title="Rank")),
            text=f"A{animal}",
            textposition="top center",
            name=f"Animal {animal} (Rank {rank})",
            showlegend=False
        ))

    fig.update_layout(
        title=f'PCA of Activity Features (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})',
        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
        height=600,
        width=800
    )

    if output_path is not None:
        fig.write_html(output_path)

    if show:
        fig.show()

    print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Variance Explained: {pca.explained_variance_ratio_.sum():.2%}")
    return fig

def plot_animals_activity(animals_protocols, ranks, output_path=None, show=True):
    # Create figure
    fig = go.Figure()

    # Add all animals
    for animal_number in animals_protocols.keys():
        if animal_number not in ranks:
            print(f"{animal_number} not in ranks, skipping")
            continue
        print(f"{animal_number} in ranks, adding to plot")

        datetime_list = animals_protocols[animal_number].data.index.tolist()
        values = animals_protocols[animal_number].data['values'].tolist()
        fig.add_trace(go.Scatter(
            x=datetime_list,
            y=values,
            mode='lines',
            name=f'Animal {animal_number} - Rank {ranks[animal_number]}' if animal_number in ranks else f'Animal {animal_number}',
            line=dict(width=1)
        ))

    fig.update_layout(
        title='Animal Activity Comparison - Only ranked',
        xaxis_title='DateTime',
        yaxis_title='Values',
        hovermode='x unified',
        height=800,
        width=1600
    )

    if output_path is not None:
        fig.write_html(output_path)

    if show:
        fig.show()
    return fig


def plot_correlation(corr_df, output_path=None, show=True):
    # Visualize top correlations
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=corr_df.index[:10],
        y=corr_df['correlation'][:10],
        marker_color=['green' if p < 0.05 else 'gray' for p in corr_df['p_value'][:10]]
    ))
    fig.update_layout(
        title='Top 10 Feature Correlations with Rank',
        xaxis_title='Feature',
        yaxis_title='Spearman Correlation',
        height=500,
        width=1200
    )

    if output_path is not None:
        fig.write_html(output_path)
    if show:
        fig.show()
    return fig


def plot_cross_correlation(animals_protocols, ranks, output_path=None, show=True):
    ## Cross-Correlation Analysis - Activity Synchronization



    # Compute pairwise cross-correlations
    cross_corr_matrix = np.zeros((len(animals_protocols), len(animals_protocols)))
    animal_list = [a for a in animals_protocols.keys() if a in ranks]

    for i, animal1 in enumerate(animal_list):
        for j, animal2 in enumerate(animal_list):
            if i <= j:
                data1 = animals_protocols[animal1].data['values'].values
                data2 = animals_protocols[animal2].data['values'].values
                
                # Normalize
                data1_norm = (data1 - data1.mean()) / data1.std()
                data2_norm = (data2 - data2.mean()) / data2.std()
                
                # Cross-correlation at zero lag
                corr = np.corrcoef(data1_norm, data2_norm)[0, 1]
                cross_corr_matrix[i, j] = corr
                cross_corr_matrix[j, i] = corr

    # Visualize correlation matrix
    fig = go.Figure(data=go.Heatmap(
        z=cross_corr_matrix,
        x=[f"A{a} (R{ranks[a]})" for a in animal_list],
        y=[f"A{a} (R{ranks[a]})" for a in animal_list],
        colorscale='RdBu',
        zmid=0,
        text=np.round(cross_corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))

    fig.update_layout(
        title='Activity Cross-Correlation Matrix',
        xaxis_title='Animal (Rank)',
        yaxis_title='Animal (Rank)',
        height=600,
        width=700
    )

    if output_path is not None:
        fig.write_html(output_path)

    if show:
        fig.show()

    # Average correlation with other animals (measure of synchronization)
    avg_corr = []
    for i, animal in enumerate(animal_list):
        others_corr = [cross_corr_matrix[i, j] for j in range(len(animal_list)) if i != j]
        avg_corr.append(np.mean(others_corr))

    sync_df = pd.DataFrame({
        'animal': animal_list,
        'avg_correlation': avg_corr,
        'actual_rank': [ranks[a] for a in animal_list]
    })
    sync_df = sync_df.sort_values('actual_rank')

    print("\nAverage Activity Synchronization:")
    print(sync_df)

    corr_sync, pval_sync = stats.spearmanr(sync_df['avg_correlation'], sync_df['actual_rank'])
    print(f"\nCorrelation between synchronization and rank: {corr_sync:.3f} (p={pval_sync:.3f})")

    return corr_sync, pval_sync, fig



def summary_visualization(all_features, output_path=None):

    ## Visual Comparison - Algorithm Ranks vs Ground Truth
    rank_comparison = all_features[['animal', 'actual_rank', 'rf_rank', 'gb_rank', 'ridge_rank', 'ada_rank', 'et_rank', 'logreg_rank']].sort_values('actual_rank')

    # Create interactive comparison plot
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Random Forest',
            'Gradient Boosting',
            'Ridge',
            'AdaBoost',
            'Extra Trees',
            'Logistic Regression'
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )

    methods = [
        ('rf_rank', 'Random Forest'),
        ('gb_rank', 'Gradient Boosting'),
        ('ridge_rank', 'Ridge'),
        ('ada_rank', 'AdaBoost'),
        ('et_rank', 'Extra Trees'),
        ('logreg_rank', 'Logistic Regression')
    ]

    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

    for (rank_col, method_name), (row, col) in zip(methods, positions):
        fig.add_trace(
            go.Scatter(
                x=rank_comparison['actual_rank'],
                y=rank_comparison[rank_col],
                mode='markers+text',
                text=[f"A{a}" for a in rank_comparison['animal']],
                textposition="top center",
                marker=dict(size=12, color=abs(rank_comparison['actual_rank'] - rank_comparison[rank_col]),
                        colorscale='RdYlGn_r', showscale=(row==1 and col==2)),
                name=method_name,
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=[1, 8],
                y=[1, 8],
                mode='lines',
                line=dict(color='black', dash='dash', width=1),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Actual Rank", row=row, col=col, range=[0, 9])
        fig.update_yaxes(title_text="Predicted Rank", row=row, col=col, range=[0, 9])

    fig.update_layout(
        title_text="Algorithm-Generated Ranks vs Ground Truth",
        height=900,
        width=1200,
        showlegend=False
    )

    if output_path is not None:
        fig.write_html(output_path)
    fig.show()

    print("✓ Visual comparison saved to ./test_results_2026/rank_comparison_visual.html")
    return fig



def generate_methods_comparison(all_features, output_folder=None, show=True):

    rank_comparison = all_features[['animal', 'actual_rank', 'rf_rank', 'gb_rank', 'ridge_rank', 'ada_rank', 'et_rank', 'logreg_rank']].sort_values('actual_rank')

    ## Accuracy Analysis - Detailed Correctness Metrics

    print("="*80)
    print("RANK PREDICTION ACCURACY ANALYSIS")
    print("="*80)

    # Calculate different accuracy metrics for each method
    methods_dict = {
        'Random Forest': 'rf_rank',
        'Gradient Boosting': 'gb_rank',
        'Ridge': 'ridge_rank',
        'AdaBoost': 'ada_rank',
        'Extra Trees': 'et_rank',
        'Logistic Regression': 'logreg_rank',
    }

    accuracy_results = []

    for method_name, rank_col in methods_dict.items():
        # Exact match accuracy (perfect prediction)
        exact_match = (rank_comparison['actual_rank'] == rank_comparison[rank_col]).sum()
        exact_match_pct = (exact_match / len(rank_comparison)) * 100
        
        # Within ±1 rank accuracy
        within_1 = (abs(rank_comparison['actual_rank'] - rank_comparison[rank_col]) <= 1).sum()
        within_1_pct = (within_1 / len(rank_comparison)) * 100
        
        # Within ±2 ranks accuracy
        within_2 = (abs(rank_comparison['actual_rank'] - rank_comparison[rank_col]) <= 2).sum()
        within_2_pct = (within_2 / len(rank_comparison)) * 100
        
        # Mean Absolute Error
        mae = abs(rank_comparison['actual_rank'] - rank_comparison[rank_col]).mean()
        
        # Spearman correlation
        corr = stats.spearmanr(rank_comparison['actual_rank'], rank_comparison[rank_col])[0]
        
        accuracy_results.append({
            'Method': method_name,
            'Exact Match (%)': exact_match_pct,
            'Within ±1 (%)': within_1_pct,
            'Within ±2 (%)': within_2_pct,
            'MAE': mae,
            'Spearman ρ': corr
        })

    accuracy_df = pd.DataFrame(accuracy_results)

    print("\nACCURACY METRICS BY METHOD:")
    print("-" * 80)
    print(accuracy_df.to_string(index=False))

    # Visualize accuracy comparison
    fig = go.Figure()

    methods = accuracy_df['Method'].tolist()
    x_pos = list(range(len(methods)))

    fig.add_trace(go.Bar(
        x=methods,
        y=accuracy_df['Exact Match (%)'],
        name='Exact Match',
        marker_color='#2E86AB'
    ))

    fig.add_trace(go.Bar(
        x=methods,
        y=accuracy_df['Within ±1 (%)'],
        name='Within ±1 Rank',
        marker_color='#A23B72'
    ))

    fig.add_trace(go.Bar(
        x=methods,
        y=accuracy_df['Within ±2 (%)'],
        name='Within ±2 Ranks',
        marker_color='#F18F01'
    ))

    fig.update_layout(
        title='Rank Prediction Accuracy by Method',
        xaxis_title='Method',
        yaxis_title='Accuracy (%)',
        barmode='group',
        height=500,
        width=1000,
        yaxis_range=[0, 100]
    )

    if output_folder is not None:
        output_path = f"{output_folder}/accuracy_comparison.html"
        fig.write_html(output_path)

    if show:
        fig.show()

    # Individual animal prediction analysis
    print("\n" + "="*80)
    print("PER-ANIMAL PREDICTION ANALYSIS:")
    print("="*80)

    for idx, row in rank_comparison.iterrows():
        animal = row['animal']
        actual = row['actual_rank']
        print(f"\nAnimal {animal} (Actual Rank: {actual}):")
        
        for method_name, rank_col in methods_dict.items():
            predicted = row[rank_col]
            error = abs(actual - predicted)
            correct = "✓ CORRECT" if error == 0 else f"✗ Error: ±{int(error)}"
            print(f"  {method_name:20s}: Rank {int(predicted)}  {correct}")

    # Best performing method
    best_method_idx = accuracy_df['Exact Match (%)'].idxmax()
    best_method = accuracy_df.loc[best_method_idx, 'Method']
    best_accuracy = accuracy_df.loc[best_method_idx, 'Exact Match (%)']

    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"Best Performing Method: {best_method}")
    print(f"Exact Match Accuracy: {best_accuracy:.1f}%")
    print(f"Within ±1 Rank Accuracy: {accuracy_df.loc[best_method_idx, 'Within ±1 (%)']:.1f}%")
    print(f"Spearman Correlation: {accuracy_df.loc[best_method_idx, 'Spearman ρ']:.3f}")

    # Save accuracy results
    if output_folder is not None:
        output_path = f"{output_folder}/accuracy_metrics.csv"
        accuracy_df.to_csv(output_path, index=False)
        print(f"\n✓ Accuracy metrics saved to {output_path}")
    return fig


def export_plots_to_pdf(figures, output_path, title=None):
    """Export a list of Plotly figures to a single multi-page PDF.

    Requires kaleido: pip install kaleido
    Each figure becomes one page in the PDF.

    Parameters
    ----------
    figures : list of (label, fig) tuples
        e.g. [('PCA', fig_pca), ('Correlation', fig_corr), ...]
    output_path : str
        Path to the output PDF file.
    title : str, optional
        Title printed on the cover page.
    """
    try:
        import kaleido  # noqa: F401
    except ImportError:
        raise ImportError("kaleido is required for PDF export. Install it with: pip install kaleido")

    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import tempfile, io

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with PdfPages(output_path) as pdf:
        # Optional cover page
        if title:
            fig_cover, ax = plt.subplots(figsize=(11, 8.5))
            ax.text(0.5, 0.55, title, ha='center', va='center',
                    fontsize=24, fontweight='bold', transform=ax.transAxes)
            ax.text(0.5, 0.42, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                    ha='center', va='center', fontsize=12, color='gray', transform=ax.transAxes)
            ax.axis('off')
            pdf.savefig(fig_cover, bbox_inches='tight')
            plt.close(fig_cover)

        for label, plotly_fig in figures:
            # Render plotly figure to PNG bytes via kaleido
            img_bytes = plotly_fig.to_image(format='png', width=1400, height=900, scale=2)
            img_buf = io.BytesIO(img_bytes)
            img = mpimg.imread(img_buf, format='png')

            fig_page, ax = plt.subplots(figsize=(14, 9))
            ax.imshow(img)
            ax.axis('off')
            if label:
                fig_page.suptitle(label, fontsize=11, y=0.02, color='gray')
            pdf.savefig(fig_page, bbox_inches='tight', dpi=150)
            plt.close(fig_page)

    print(f"✓ PDF saved to {output_path}  ({len(figures)} pages)")