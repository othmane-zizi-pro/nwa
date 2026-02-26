# Requirement-to-Evidence Mapping

## Final Project Deliverables
- `deliverables/final_presentation.pptx`: final slide deck
- `deliverables/repo_link.md`: repository link + student names/IDs
- `deliverables/presentation_notes.tex`: notes prepared in LaTeX
- `deliverables/presentation_notes.pdf`: notes PDF generated from LaTeX

## Section 5 Data Science Lifecycle
- `5.1 Framing`: `deliverables/final_presentation.pptx` (Context/Hypothesis slides)
- `5.2 Data Acquisition`: `src/data_loader.py` (`ensure_raw_data`)
- `5.3 Data Exploration`: `notebooks/01_data_exploration.ipynb`
- `5.4 Data Preparation`: `notebooks/02_data_cleaning.ipynb`
- `5.5 Modeling`: `notebooks/04_modeling.ipynb`, `src/models.py`
- `5.6 Model Evaluation`: `reports/model_results.csv`, `reports/cv_results.csv`
- `5.7 Model Selection`: `scripts/run_full_pipeline.py` best-model logic
- `5.8 Fine-Tuning`: `scripts/run_full_pipeline.py`, `reports/tuning_results.csv`
- `5.9 Solution Presentation`: `deliverables/final_presentation.pptx`
- `5.10 Launching/Monitoring/Maintenance`: `reports/monitoring_plan.md`, `reports/monitoring_input_drift.csv`

## Engineering Quality
- Unit tests: `tests/test_features.py`, `tests/test_monitoring.py`
- Validation command: `python3 -m pytest -q`

## Explainability
- Feature importance: `reports/feature_importance.csv`, `reports/figures/feature_importance.png`
- SHAP: `reports/figures/shap_summary.png`

## CausalML Phase
- Target/Treatment/Controls: `notebooks/06_causal_inference.ipynb`, `src/causal.py`
- Required regressors: `LRSRegressor`, `XGBTRegressor` in `src/causal.py`
- Causal outputs: `reports/causal_ate_results.csv`, `reports/causal_feature_importance.csv`
