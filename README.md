# Same Vitals, Different Scores
### Language-Minority Undertriage at the ESI 3 Decision Boundary

A LightGBM-based triage acuity prediction model with bias analysis across language groups in a synthetic Finnish emergency department dataset.

Submitted to the [Triagegeist Competition](https://kaggle.com/competitions/triagegeist) hosted by the Laitinen-Fredriksson Foundation.

---

## Repository Structure

```
triagegeist/
├── src/
│   ├── charts.py                      # Chart generation for bias analysis
│   ├── config.py                      # Parameters and constants
│   ├── feature_importance_analysis.py # Feature importance extraction
│   ├── model_predictions.py           # Prediction pipeline
│   ├── model_training.py              # Training and cross-validation
│   ├── statistical_testing.py         # Chi-square and Mann-Whitney tests
│   ├── tfidf_model.py                 # NLP experiment (TF-IDF)
│   └── utils.py                       # Data loading utilities
├── assets/                            # Charts and figures for writeup
├── notebooks/
│   └── training-pipeline.ipynb        # Main notebook — run this
└── README.md
```

---

## Reproducing Results

### 1. Dataset Setup

This notebook uses the Triagegeist synthetic dataset provided by the competition. Before running, ensure the following files are placed in `/kaggle/input/triagegeist/`:

```
train.csv
test.csv
chief_complaints.csv
patient_history.csv
sample_submission.csv
```

### 2. Run the Notebook

Once the dataset is in place, run `training-pipeline.ipynb` cell by cell from top to bottom or run all the cells in one go. The Kaggle notebook clones this repository automatically and imports the relevant modules which call the training pipeline and model predictions. No GPU is required. Expected runtime is approximately 5-10 minutes on a standard Kaggle CPU environment.

### 3. Output

The notebook produces `submission.csv` in the working directory, containing `patient_id` and predicted `triage_acuity` (ESI 1-5) for all 20,000 test patients.

---

## Reproducibility

| Component | Value |
|---|---|
| Random seed | 42 (all stochastic components) |
| CV strategy | Stratified 5-fold |
| Environment | Kaggle CPU, pinned 2026-02-19 |
| GPU required | No |

---

## Key Findings

- **QWK 0.9309 ± 0.0009** across five stratified folds, exceeding published human inter-rater benchmarks (0.6-0.8)
- **ESI 1+2 sensitivity 98.1%** — all misclassifications confined to adjacent ESI levels
- **Arabic-speaking patients are undertriaged at a statistically significant rate at ESI 3** (12.0% vs 10.0% for Finnish speakers, chi-square p=0.019) despite equivalent NEWS2 scores (Mann-Whitney p=0.360)

---

## License

For competition purposes only. Dataset provided by the Laitinen-Fredriksson Foundation under competition terms.