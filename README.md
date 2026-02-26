# Customer Lifetime Value Prediction (INSY 674 Final Project)

Repository: https://github.com/othmane-zizi-pro/nwa

## Team
| Member | Student ID | GitHub |
|---|---:|---|
| Othmane Zizi | 261255341 | `othmane-zizi-pro` |
| Fares Joni | 261254593 | `FaresJ81` |
| Tanmay Giri | 261272443 | `tanmaysgiri` |

## Project Summary
End-to-end ML pipeline for 6-month CLV prediction, explainability (feature importance + SHAP), causal inference (CausalML), and monitoring artifacts.

## Structure
```text
nwa/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_segmentation.ipynb
│   ├── 06_causal_inference.ipynb
│   └── 07_launch_monitoring.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── causal.py
│   └── monitoring.py
├── scripts/
│   └── run_full_pipeline.py
├── tests/
│   ├── test_features.py
│   └── test_monitoring.py
├── reports/
├── deliverables/
│   ├── final_presentation.pptx
│   ├── repo_link.md
│   ├── presentation_notes.tex
│   ├── presentation_notes.pdf
│   └── requirements_mapping.md
└── data/
```

## Setup
```bash
python3 -m pip install -r requirements.txt
```

## Run End-to-End Pipeline
```bash
python3 scripts/run_full_pipeline.py
```

This generates:
- `data/processed/cleaned_retail.csv`
- `data/processed/customer_features.csv`
- `data/processed/clv_predictions.csv`
- `data/processed/customer_segments.csv`
- `data/processed/causal_dataset.csv`
- `reports/*.csv`
- `reports/figures/*.png`

## Execute All Notebooks (with outputs)
```bash
for nb in notebooks/01_data_exploration.ipynb notebooks/02_data_cleaning.ipynb notebooks/03_feature_engineering.ipynb notebooks/04_modeling.ipynb notebooks/05_segmentation.ipynb notebooks/06_causal_inference.ipynb notebooks/07_launch_monitoring.ipynb; do
  python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb"
done
```

## Run Tests
```bash
python3 -m pytest -q
```

## External Storage Placeholders (Large Files)
Replace these links before submission:
- Raw data folder: `[LINK_TO_RAW_DATA_FOLDER]`
- Processed data folder: `[LINK_TO_PROCESSED_DATA_FOLDER]`
- Reports/artifacts folder: `[LINK_TO_REPORTS_FOLDER]`

Upload to storage:

`raw`
- `data/raw/online_retail_II.xlsx`
- `data/raw/README_data_source.txt` (source URL + download date + checksum)

`processed`
- `data/processed/cleaned_retail.csv`
- `data/processed/customer_features.csv`
- `data/processed/clv_predictions.csv`
- `data/processed/customer_segments.csv`
- `data/processed/causal_dataset.csv`

`reports`
- `reports/model_results.csv`
- `reports/cv_results.csv`
- `reports/feature_importance.csv`
- `reports/causal_ate_results.csv`
- `reports/causal_feature_importance.csv`
- `reports/causal_uplift_predictions.csv`
- `reports/monitoring_input_drift.csv`
- `reports/monitoring_plan.md`
- `reports/figures/` (all generated PNGs)

## Notes
- SHAP output: `reports/figures/shap_summary.png`
- Feature importance: `reports/feature_importance.csv` and `reports/figures/feature_importance.png`
- Causal models required by assignment: `LRSRegressor`, `XGBTRegressor` (see notebook `06`)
