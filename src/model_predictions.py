"""Model predictions without linguistic processing"""

import pandas as pd
import numpy as np
from numpy.typing import NDArray
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from utils import read_datasets
from config import (
    params,
    num_boost_round,
    early_stopping,
    log_evaluation,
    missing_flag_cols,
    filepath,
    drop_cols,
)


def clean_dataframe(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Clean dataframe"""

    # Apply same preprocessing as train
    df['pain_score'] = df['pain_score'].replace(-1, np.nan)

    for col in missing_flag_cols:
        df[f'{col}_missing'] = df[col].isna().astype(int)

    df['gcs_abnormal'] = (df['gcs_total'] < 15).astype(int)

    # Align feature columns with training
    drop_cols_test = ['patient_id', 'bmi']
    feature_cols_test = [c for c in feature_cols if c in df.columns]

    # Encode categoricals
    cat_cols = df[feature_cols_test].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    X_test = df[feature_cols_test]

    return X_test, cat_cols

def cross_validation(
        X_test: pd.DataFrame, 
        X: pd.DataFrame, 
        y: pd.Series, 
        cat_cols: list[str],
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> NDArray[np.float64]:
    """Perform n-fold cross validation"""

    # Retrain on full training data with 5-fold and average probabilities
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    test_probs = np.zeros((len(X_test), 5))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train_fold, label=y_train_fold, categorical_feature=cat_cols)
        dval = lgb.Dataset(X_val_fold, label=y_val_fold, categorical_feature=cat_cols)

        model_fold = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(-1)]
        )

        fold_probs = model_fold.predict(X_test)
        test_probs += fold_probs / n_splits
        print(f"Fold {fold+1} done")
    
    return test_probs


def run_prediction_pipeline(filepath: str) -> None:
    """Run prediction pipeline"""

    test = read_datasets(f'{filepath}/test.csv')
    test_history = read_datasets(f'{filepath}/patient_history.csv')
    test_df = test.merge(test_history, on='patient_id', how='left')

    train = read_datasets(f'{filepath}/train.csv')
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X_test, cat_cols = clean_dataframe(test_df, feature_cols)

    X = train[feature_cols]
    y = train['triage_acuity'] - 1

    test_probs = cross_validation(X_test, X, y, cat_cols)

    # Final predictions — average probabilities across folds then argmax
    test_preds = np.argmax(test_probs, axis=1) + 1  # back to 1-indexed ESI

    # Create submission
    submission = read_datasets(f'{filepath}/sample_submission.csv')
    submission['triage_acuity'] = test_preds

    print("\nPrediction distribution:")
    print(pd.Series(test_preds).value_counts().sort_index())

    print("\nSample submission:")
    print(submission.head())

    submission.to_csv('submission.csv', index=False)
    print("\nSaved to submission.csv")