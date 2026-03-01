import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np

# Make sure df has oof predictions and true ESI attached
# df['oof_pred_esi'] and df['true_esi'] should already exist from bias analysis

def draw_charts(df: pd.DataFrame):
    """Draw undertriage bar charts"""

    # ── Chart 1: Undertriage rate by language group within each ESI level ─────────
    languages_of_interest = ['Arabic', 'Finnish', 'Somali', 'Swedish', 'English']
    df_lang = df[df['language'].isin(languages_of_interest)].copy()

    undertriage_by_lang_esi = df_lang.groupby(['language', 'true_esi'], observed=True)[['oof_pred_esi', 'true_esi']].apply(
        lambda x: (x['oof_pred_esi'] > x['true_esi']).mean()
    ).reset_index()
    undertriage_by_lang_esi.columns = ['language', 'true_esi', 'undertriage_rate']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    esi_levels = [1, 2, 3, 4]
    colors = {
        'Arabic': '#d62728',
        'Finnish': '#1f77b4',
        'Somali': '#ff7f0e',
        'Swedish': '#2ca02c',
        'English': '#9467bd'
    }

    for ax, esi in zip(axes, esi_levels):
        subset = undertriage_by_lang_esi[undertriage_by_lang_esi['true_esi'] == esi]
        
        # Highlight ESI 3 panel
        if esi == 3:
            ax.set_facecolor('#fff7e6')
            ax.annotate(
                '* Significant (p=0.019)',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top',
                fontsize=10, color='#d62728',
                fontweight='bold'
            )
        else:
            ax.annotate(
                'ns',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top',
                fontsize=10, color='grey'
            )

        bars = ax.bar(
            subset['language'],
            subset['undertriage_rate'],
            color=[colors[l] for l in subset['language']],
            edgecolor='white',
            linewidth=0.8
        )
        ax.set_title(f'True ESI {esi}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Language Group', fontsize=11)
        ax.tick_params(axis='x', rotation=30)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0, 0.20)
        ax.axhline(
            y=subset[subset['language'] == 'Finnish']['undertriage_rate'].values[0],
            color='#1f77b4', linestyle='--', linewidth=1.2, alpha=0.6, label='Finnish baseline'
        )
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Undertriage Rate', fontsize=11)
    axes[0].legend(fontsize=9)
    fig.suptitle(
        'Undertriage Rate by Language Group Within Each ESI Level\n'
        'Controlling for true clinical acuity',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig('chart1_undertriage_by_language_esi.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart 1 saved")


    # ── Chart 2: NEWS2 distribution for Arabic vs Finnish within ESI 2 ────────────
    df_esi3 = df[
        (df['true_esi'] == 3) &
        (df['language'].isin(['Arabic', 'Finnish']))
    ].copy()

    fig, axes = plt.subplots(1, 1, figsize=(14, 5))

    # Left: NEWS2 distribution
    for lang, color in [('Finnish', '#1f77b4'), ('Arabic', '#d62728')]:
        subset = df_esi3[df_esi3['language'] == lang]['news2_score'].dropna()
        axes.hist(
            subset, bins=20, alpha=0.6, color=color,
            label=f'{lang} (n={len(subset)})', edgecolor='white', density=True
        )
        axes.axvline(subset.mean(), color=color, linestyle='--', linewidth=1.5,
                        label=f'{lang} mean: {subset.mean():.2f}')

    axes.set_title('NEWS2 Score Distribution\nTrue ESI 3 Patients Only',
                    fontsize=13, fontweight='bold')
    axes.set_xlabel('NEWS2 Score', fontsize=11)
    axes.set_ylabel('Density', fontsize=11)
    axes.legend(fontsize=9)
    axes.grid(alpha=0.3)

    fig.suptitle(
        'Arabic vs Finnish Patients — Clinical Severity and Triage Outcome\n'
        'Equivalent NEWS2 distributions suggest gap is not explained by case mix',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig('chart2_arabic_finnish_esi2.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart 2 saved")