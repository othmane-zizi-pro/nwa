"""Monitoring utilities for launch/maintenance deliverables."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions."""
    expected = expected.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    actual = actual.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if expected.empty or actual.empty:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(expected, quantiles))
    if len(breaks) <= 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breaks)
    actual_counts, _ = np.histogram(actual, bins=breaks)

    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def build_input_drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature-level input drift report using PSI."""
    common_cols = [col for col in baseline_df.columns if col in current_df.columns]
    rows = []
    for col in common_cols:
        psi = population_stability_index(baseline_df[col], current_df[col])
        if np.isnan(psi):
            level = "unknown"
        elif psi < 0.1:
            level = "stable"
        elif psi < 0.25:
            level = "moderate_drift"
        else:
            level = "high_drift"
        rows.append({"Feature": col, "PSI": psi, "DriftLevel": level})

    return pd.DataFrame(rows).sort_values("PSI", ascending=False)


def generate_monitoring_plan(
    output_path: Path,
    model_name: str,
    primary_metric: str = "RMSE",
) -> None:
    """Create a concise monitoring + maintenance plan aligned with section 5.10."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Launch, Monitoring, and Maintenance Plan

## Productionization Checklist
- Register and version final model artifact: `{model_name}`
- Freeze feature schema and enforce validation checks before scoring
- Add unit tests for preprocessing, feature generation, and scoring paths
- Enable batch scoring pipeline for weekly CLV refresh

## Monitoring
- Model quality: monitor `{primary_metric}` weekly on latest labeled cohort
- Business KPI: track CLV captured by top 20% ranked customers
- Input quality: null-rate and type checks per feature
- Drift detection: PSI on each feature (alert if PSI >= 0.25)

## Alerting Thresholds
- Warning: `{primary_metric}` deteriorates by >=10% vs. baseline
- Critical: `{primary_metric}` deteriorates by >=20% vs. baseline
- Critical: any core feature with PSI >= 0.25 for 2 consecutive runs

## Maintenance and Retraining
- Scheduled retraining: monthly
- Emergency retraining: triggered by critical alerts
- Keep last 3 production model versions for rollback
- Re-run explainability (feature importance + SHAP) after each retrain
"""
    output_path.write_text(content, encoding="utf-8")


def create_monitoring_artifacts(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> Dict[str, Path]:
    """Generate monitoring artifacts required for submission."""
    output_dir.mkdir(parents=True, exist_ok=True)
    drift_report = build_input_drift_report(baseline_df, current_df)
    drift_path = output_dir / "monitoring_input_drift.csv"
    drift_report.to_csv(drift_path, index=False)

    plan_path = output_dir / "monitoring_plan.md"
    generate_monitoring_plan(plan_path, model_name=model_name)

    return {
        "drift_report": drift_path,
        "monitoring_plan": plan_path,
    }
