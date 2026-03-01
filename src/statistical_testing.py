from scipy import stats
import pandas as pd

def perform_statistical_testing(df: pd.DataFrame) -> None:
    """Perform MannWhitneyu and ChiSquare tests"""
    df_esi3 = df[
        (df['true_esi'] == 3) &
        (df['language'].isin(['Arabic', 'Finnish']))
    ].copy()

    # ── Mann-Whitney U: NEWS2 between Arabic and Finnish ESI 2 patients ───────────
    arabic_news2_esi3 = df_esi3[df_esi3['language'] == 'Arabic']['news2_score'].dropna()
    finnish_news2_esi3 = df_esi3[df_esi3['language'] == 'Finnish']['news2_score'].dropna()

    stat, p_value = stats.mannwhitneyu(arabic_news2_esi3, finnish_news2_esi3, alternative='two-sided')

    print("=== NEWS2 Clinical Severity Comparison (ESI 3 patients) ===")
    print(f"Arabic  — n={len(arabic_news2_esi3)}, mean={arabic_news2_esi3.mean():.3f}, median={arabic_news2_esi3.median():.3f}")
    print(f"Finnish — n={len(finnish_news2_esi3)}, mean={finnish_news2_esi3.mean():.3f}, median={finnish_news2_esi3.median():.3f}")
    print(f"Mann-Whitney U statistic: {stat:.1f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Interpretation: {'No significant difference in clinical severity' if p_value > 0.05 else 'Significant difference in clinical severity'}")

    # ── Chi-square: Undertriage rates by language within each ESI level ───────────
    print("\n=== Undertriage Rate Comparison: Arabic vs Finnish by ESI Level ===")
    results = []

    for esi in [1, 2, 3, 4]:
        subset = df[
            (df['true_esi'] == esi) &
            (df['language'].isin(['Arabic', 'Finnish']))
        ].copy()
        subset['undertriaged'] = (subset['oof_pred_esi'] > subset['true_esi']).astype(int)

        arabic = subset[subset['language'] == 'Arabic']
        finnish = subset[subset['language'] == 'Finnish']

        # Contingency table: rows = language, cols = undertriaged/not
        contingency = pd.crosstab(subset['language'], subset['undertriaged'])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)

        arabic_rate = arabic['undertriaged'].mean()
        finnish_rate = finnish['undertriaged'].mean()

        results.append({
            'ESI Level': esi,
            'Arabic n': len(arabic),
            'Finnish n': len(finnish),
            'Arabic undertriage %': f"{arabic_rate:.1%}",
            'Finnish undertriage %': f"{finnish_rate:.1%}",
            'Chi2': round(chi2, 3),
            'p-value': round(p, 4),
            'Significant': 'Yes' if p < 0.05 else 'No'
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))