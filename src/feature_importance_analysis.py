"""Feature importance analysis"""

import pandas as pd
import numpy as np
import lightgbm as lgb

from config import params, num_boost_round


def find_important_and_unimportant_features(
        X: pd.DataFrame, 
        y: pd.Series, 
        cat_cols: list[str], 
        n_top: int = 20, 
        n_bottom: int = 10
    ) -> None:
    """Find top n and bottom n features"""
    # Retrain on full data to get stable feature importance
    dtrain_full = lgb.Dataset(X, label=y, categorical_feature=cat_cols)

    model_full = lgb.train(
        params,
        dtrain_full,
        num_boost_round=num_boost_round,
        callbacks=[lgb.log_evaluation(-1)]
    )

    # Feature importance — both split and gain
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_split': model_full.feature_importance(importance_type='split'),
        'importance_gain': model_full.feature_importance(importance_type='gain')
    })

    importance_df = importance_df.sort_values('importance_gain', ascending=False)
    print("Top 20 features by gain:")
    print(importance_df.head(n_top).to_string(index=False))

    print("\nBottom 10 features by gain:")
    print(importance_df.tail(n_bottom).to_string(index=False))