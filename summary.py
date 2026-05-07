## Summary Report - Rank Inference Results
def generate_summary_report(X_scaled, all_features, y, corr_df, rf, mae, corr_sync, pval_sync, X_topk=None, output_path=None):
    lines = []
    lines.append("="*80)
    lines.append("RANK INFERENCE ANALYSIS SUMMARY")
    lines.append("="*80)

    lines.append("\n1. BASIC STATISTICS")
    lines.append(f"   Number of animals analyzed: {len(all_features)}")
    lines.append(f"   Rank range: {all_features['actual_rank'].min()} to {all_features['actual_rank'].max()}")

    lines.append("\n2. TOP PREDICTIVE FEATURES (by absolute correlation):")
    for idx, row in corr_df.head(5).iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        lines.append(f"   {idx:25s}: {row['correlation']:+.3f} (p={row['p_value']:.4f}) {sig}")

    lines.append("\n5. SUPERVISED LEARNING (Random Forest)")
    lines.append(f"   Mean Absolute Error (Leave-One-Out CV): {mae:.2f} ranks")
    X_for_score = X_topk if X_topk is not None else X_scaled
    lines.append(f"   R² Score: {rf.score(X_for_score, y):.3f}")

    lines.append("\n6. ACTIVITY SYNCHRONIZATION")
    lines.append(f"   Correlation with rank: {corr_sync:.3f} (p={pval_sync:.3f})")

    lines.append("\n7. PREDICTED vs ACTUAL RANKS:")
    comparison = all_features[['animal', 'actual_rank', 'rf_rank']].sort_values('actual_rank')
    comparison['rank_error'] = abs(comparison['actual_rank'] - comparison['rf_rank'])
    lines.append(comparison.to_string(index=False))
    lines.append(f"\n   Average prediction error: {comparison['rank_error'].mean():.2f} ranks")

    lines.append("\n" + "="*80)
    lines.append("All results have been saved to ./test_results_2026/")
    lines.append("="*80)

    summary_text = "\n".join(lines)
    print(summary_text)

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(summary_text)
        print(f"Saved summary report to {output_path}")

