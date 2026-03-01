

# source filepath
filepath = "/kaggle/working"

# source fields
missing_flag_cols = ['systolic_bp', 'diastolic_bp', 'mean_arterial_pressure', 
                     'pulse_pressure', 'shock_index', 'respiratory_rate']

drop_cols = ['patient_id', 'triage_acuity', 'disposition', 'ed_los_hours', 'bmi']
# LGBM params

num_boost_round = 500
early_stopping = 50
log_evaluation = 100
params = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
