"""tF-IDF implementation"""

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb

from utils import read_datasets
from config import (
    params, 
    num_boost_round, 
    early_stopping,
    log_evaluation,
)


# Load complaints and join
complaints = pd.read_csv(f'{path}/chief_complaints.csv')
df2 = df.merge(complaints, on='patient_id', how='left')
df2['chief_complaint_raw'] = df2['chief_complaint_raw'].fillna('unknown')

# Drop same columns as before plus chief_complaint_system since it's in structured features
# drop_cols = ['patient_id', 'triage_acuity', 'disposition', 'ed_los_hours', 
#              'bmi', 'chief_complaint_raw']

def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, NDArray[str], list[str]]:
    """Prepare dataframe before training"""
    drop_cols = [
        'patient_id', 'triage_acuity', 'disposition', 'ed_los_hours',
        'bmi', 'chief_complaint_raw',
        'chief_complaint_system_y',  # duplicate from merge
        # bias analysis columns that leaked in from df
        'oof_pred', 'oof_pred_esi', 'true_esi', 
        'undertriage', 'overtriage', 'correct'
    ]
    feature_cols2 = [c for c in df.columns if c not in drop_cols]

    # Re-identify categoricals
    cat_cols2 = df[feature_cols2].select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols2:
        df[col] = df[col].astype('category')

    X2 = df[feature_cols2]
    y2 = df['triage_acuity'] - 1
    text = df['chief_complaint_raw'].values

    return X2, y2, text, cat_cols2

def cross_validation(
        X: pd.DataFrame,
        y: pd.Series,
        text: NDArray[str],
        cat_cols: list[str],
        n_splits: int = 5, 
        shuffle: bool = True, 
        random_state: int = 42
    ) -> tuple[list[float], list[float]]:

    # CV loop with TF-IDF fit inside fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    oof_preds_nlp = np.zeros(len(X))
    qwk_scores_nlp = []
    f1_scores_nlp = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        text_train, text_val = text[train_idx], text[val_idx]

        # Encode categoricals as integer codes before converting to sparse
        X_train_encoded = X_train.copy()
        X_val_encoded = X_val.copy()
        
        for col in cat_cols:
            if col in X_train_encoded.columns:
                X_train_encoded[col] = X_train_encoded[col].cat.codes
                X_val_encoded[col] = X_val_encoded[col].cat.codes

        X_train_sparse = csr_matrix(X_train_encoded.to_numpy(dtype=float, na_value=np.nan))
        X_val_sparse = csr_matrix(X_val_encoded.to_numpy(dtype=float, na_value=np.nan))

        # Fit TF-IDF only on training fold
        tfidf = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            min_df=5,
            sublinear_tf=True
        )
        tfidf_train = tfidf.fit_transform(text_train)
        tfidf_val = tfidf.transform(text_val)

        X_train_combined = hstack([X_train_sparse, tfidf_train])
        X_val_combined = hstack([X_val_sparse, tfidf_val])

        dtrain = lgb.Dataset(X_train_combined, label=y_train)
        dval = lgb.Dataset(X_val_combined, label=y_val)

        model_nlp = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(log_evaluation)]
        )

        val_probs = model_nlp.predict(X_val_combined)
        val_preds = np.argmax(val_probs, axis=1)
        oof_preds_nlp[val_idx] = val_preds

        qwk = cohen_kappa_score(y_val, val_preds, weights='quadratic')
        wf1 = f1_score(y_val, val_preds, average='weighted')
        qwk_scores_nlp.append(qwk)
        f1_scores_nlp.append(wf1)
        print(f"Fold {fold+1} — QWK: {qwk:.4f} | Weighted F1: {wf1:.4f}")
    
    return qwk_scores_nlp, f1_scores_nlp


def run_tfidf_pipeline(filepath: str):
    """
    Run the TF-IDF pipeline
    1) Read the datasets
    2) Perform n-fold cross validation
    3) Train model
    """

    train = read_datasets(f'{filepath}/train.csv')
    history = read_datasets(f'{filepath}/patient_history.csv')
    df = train.merge(history, on='patient_id')
    complaints = read_datasets(f"{filepath}/chief_complaints.csv")

    df2 = df.merge(complaints, on='patient_id', how='left')
    df2['chief_complaint_raw'] = df2['chief_complaint_raw'].fillna('unknown')

    X, y, text, cat_cols = clean_dataframe(df2)

    qwk_scores_nlp, f1_scores_nlp = cross_validation(X, y, text, cat_cols)

    print(f"\nNLP Model — Mean QWK:        {np.mean(qwk_scores_nlp):.4f} ± {np.std(qwk_scores_nlp):.4f}")
    print(f"NLP Model — Mean Weighted F1: {np.mean(f1_scores_nlp):.4f} ± {np.std(f1_scores_nlp):.4f}")
    print(f"\nBaseline  — Mean QWK:        0.9309 ± 0.0009")
    print(f"Baseline  — Mean Weighted F1: 0.8569 ± 0.0018")