"""Analysis Bias across multiple demographic dimensions"""

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import cohen_kappa_score


def bias_analysis(df: pd.DataFrame, y: pd.Series, oof_preds: NDArray[np.float64]) -> None:
    """Perform Bias analysis across multiple demographic dimensions"""
    
    # Rebuild df with oof_preds attached
    df['oof_pred'] = oof_preds
    df['oof_pred_esi'] = oof_preds + 1  # back to 1-indexed
    df['true_esi'] = y + 1

    # Helper: undertriage = predicted acuity lower urgency than true (pred > true in ESI scale)
    df['undertriage'] = (df['oof_pred_esi'] > df['true_esi']).astype(int)
    df['overtriage'] = (df['oof_pred_esi'] < df['true_esi']).astype(int)
    df['correct'] = (df['oof_pred_esi'] == df['true_esi']).astype(int)

    # ── 1. Nurse-level bias ──────────────────────────────────────────────────────
    print("=== NURSE-LEVEL BIAS ===")
    nurse_stats = df.groupby('triage_nurse_id').agg(
        n=('true_esi', 'count'),
        mean_true_esi=('true_esi', 'mean'),
        mean_pred_esi=('oof_pred_esi', 'mean'),
        undertriage_rate=('undertriage', 'mean'),
        qwk=('true_esi', lambda x: cohen_kappa_score(
            x, df.loc[x.index, 'oof_pred_esi'], weights='quadratic'))
    ).round(4)

    nurse_stats['esi_bias'] = (nurse_stats['mean_pred_esi'] - nurse_stats['mean_true_esi']).round(4)
    nurse_stats = nurse_stats.sort_values('mean_true_esi')
    print(nurse_stats.to_string())

    # ── 2. Dementia undertriage ──────────────────────────────────────────────────
    print("\n=== DEMENTIA UNDERTRIAGE ===")
    for flag in [0, 1]:
        subset = df[df['hx_dementia'] == flag]
        label = 'No dementia' if flag == 0 else 'Has dementia'
        print(f"\n{label} (n={len(subset)}):")
        print(f"  Undertriage rate:     {subset['undertriage'].mean():.4f}")
        print(f"  Overtriage rate:      {subset['overtriage'].mean():.4f}")
        print(f"  Mean true ESI:        {subset['true_esi'].mean():.4f}")
        print(f"  Mean pred ESI:        {subset['oof_pred_esi'].mean():.4f}")
        print(f"  QWK:                  {cohen_kappa_score(subset['true_esi'], subset['oof_pred_esi'], weights='quadratic'):.4f}")

    # ── 3. Demographic bias ──────────────────────────────────────────────────────
    print("\n=== DEMOGRAPHIC BIAS ===")
    for col in ['sex', 'age_group', 'language', 'insurance_type']:
        print(f"\n--- {col} ---")
        stats = df.groupby(col).agg(
            n=('true_esi', 'count'),
            undertriage_rate=('undertriage', 'mean'),
            mean_true_esi=('true_esi', 'mean'),
            mean_pred_esi=('oof_pred_esi', 'mean')
        ).round(4)
        stats['esi_bias'] = (stats['mean_pred_esi'] - stats['mean_true_esi']).round(4)
        print(stats.to_string())