"""Base model"""

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, f1_score
import lightgbm as lgb

from config import (
    params, 
    num_boost_round, 
    early_stopping,
    log_evaluation,
    drop_cols,
)

from utils import read_datasets

def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Perform the following steps:
     1) Impute missing values
     2) Drop irrevalent columns 
     3) Feature engineer new columns
     4) Identify cat columns
    """
    df['pain_score'] = df['pain_score'].replace(-1, np.nan)

    # Missingness indicators
    missing_flag_cols = ['systolic_bp', 'diastolic_bp', 'mean_arterial_pressure', 
                        'pulse_pressure', 'shock_index', 'respiratory_rate']
    for col in missing_flag_cols:
        df[f'{col}_missing'] = df[col].isna().astype(int)

    # Feature engineering
    df['gcs_abnormal'] = (df['gcs_total'] < 15).astype(int)

    # Drop leaky and useless columns
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Identify categoricals for LightGBM
    cat_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype('category')
    
    X = df[feature_cols]
    y = df['triage_acuity'] - 1  # 0-indexed for LightGBM


    return X, y, cat_cols

def cross_validation(
        X: pd.DataFrame,
        y: pd.Series,
        cat_cols: list[str],
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> NDArray[np.float64]:
    """Perform n-fold cross validation"""

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    oof_preds = np.zeros(len(X))
    qwk_scores = []
    f1_scores = []

    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(log_evaluation)]
        )

        val_probs = model.predict(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        oof_preds[val_idx] = val_preds

        qwk = cohen_kappa_score(y_val, val_preds, weights='quadratic')
        wf1 = f1_score(y_val, val_preds, average='weighted')
        qwk_scores.append(qwk)
        f1_scores.append(wf1)
        print(f"Fold {fold+1} — QWK: {qwk:.4f} | Weighted F1: {wf1:.4f}")
    
    return oof_preds

def confusion_matrix(y: pd.Series, oof_preds: NDArray[np.float64], n_classes: int = 5):
    """Confusion matrix on OOF predictions"""

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, oof_preds)
    cm_df = pd.DataFrame(cm, 
                        index=[f'True ESI {i+1}' for i in range(n_classes)],
                        columns=[f'Pred ESI {i+1}' for i in range(n_classes)])
    print("\nOOF Confusion Matrix:")
    print(cm_df)

    # ESI 1+2 sensitivity
    esi12_mask = y <= 1
    esi12_sensitivity = (oof_preds[esi12_mask] <= 1).mean()
    print(f"\nESI 1+2 Sensitivity (undertriage detection): {esi12_sensitivity:.4f}")

def run_model_pipeline(
        filepath: str
) -> tuple[pd.DataFrame, pd.Series, NDArray[np.float64], list[str]]:
    """Run the model pipeline"""

    train = read_datasets(f'{filepath}/train.csv')
    history = read_datasets(f'{filepath}/patient_history.csv')
    df = train.merge(history, on='patient_id')

    X, y, cat_cols = clean_dataframe(df)

    oof_preds = cross_validation(X, y, cat_cols)

    confusion_matrix(y, oof_preds)
    return X, y, oof_preds, cat_cols