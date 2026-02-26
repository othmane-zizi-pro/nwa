"""Causal inference helpers using CausalML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from causalml.inference.meta import LRSRegressor, XGBTRegressor

    CAUSALML_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    LRSRegressor = None
    XGBTRegressor = None
    CAUSALML_AVAILABLE = False


@dataclass
class CausalRunOutput:
    ate_results: pd.DataFrame
    uplift_predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def create_campaign_treatment(
    df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a plausible treatment assignment signal for causal modeling.

    Since Online Retail II has no explicit intervention column, we emulate an
    observational campaign assignment policy based on customer behavior.
    """
    required = ["Recency", "Frequency", "Monetary", "Tenure", "NumUniqueProducts"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for treatment creation: {missing}")

    rng = np.random.default_rng(random_state)
    work = df.copy()

    z_recency = (work["Recency"] - work["Recency"].mean()) / (work["Recency"].std() + 1e-9)
    z_frequency = (work["Frequency"] - work["Frequency"].mean()) / (
        work["Frequency"].std() + 1e-9
    )
    z_monetary = (work["Monetary"] - work["Monetary"].mean()) / (work["Monetary"].std() + 1e-9)
    z_products = (work["NumUniqueProducts"] - work["NumUniqueProducts"].mean()) / (
        work["NumUniqueProducts"].std() + 1e-9
    )
    z_tenure = (work["Tenure"] - work["Tenure"].mean()) / (work["Tenure"].std() + 1e-9)

    logits = (
        0.9 * z_recency
        - 0.7 * z_frequency
        - 0.5 * z_monetary
        - 0.2 * z_products
        - 0.1 * z_tenure
        + rng.normal(0, 0.45, size=len(work))
    )
    propensity = _sigmoid(logits)
    treatment_binary = rng.binomial(1, propensity)
    treatment_group = np.where(treatment_binary == 1, "campaign", "control")

    work["TreatmentBinary"] = treatment_binary
    work["TreatmentGroup"] = treatment_group
    work["PropensityScore"] = propensity

    return work


def _extract_ate_tuple(raw_ate: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float, float, float]:
    ate, lb, ub = raw_ate
    return float(np.ravel(ate)[0]), float(np.ravel(lb)[0]), float(np.ravel(ub)[0])


def run_causal_analysis(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_col: str = "CLV",
    output_dir: Optional[Path] = None,
    control_name: str = "control",
    random_state: int = 42,
) -> CausalRunOutput:
    """
    Run CausalML regressors (LRSRegressor + XGBTRegressor).

    Outputs:
    - ATE table
    - Individual uplift predictions
    - Feature importance table
    """
    if not CAUSALML_AVAILABLE:
        raise ImportError(
            "CausalML is not available in this environment. Install `causalml` first."
        )

    if "TreatmentGroup" not in df.columns:
        raise ValueError("Input dataframe must include 'TreatmentGroup'.")

    work = df.copy()
    work[outcome_col] = work[outcome_col].astype(float)
    work["OutcomeLog1p"] = np.log1p(work[outcome_col].clip(lower=0))

    X = work[feature_cols].fillna(0).to_numpy()
    y = work["OutcomeLog1p"].to_numpy()
    treatment = work["TreatmentGroup"].to_numpy()

    lrs = LRSRegressor(control_name=control_name)
    ate_lr = _extract_ate_tuple(lrs.estimate_ate(X, treatment, y))
    cate_lr = lrs.fit_predict(X, treatment, y).ravel()

    xgbt = XGBTRegressor(control_name=control_name, random_state=random_state)
    ate_xgb = _extract_ate_tuple(xgbt.estimate_ate(X, treatment, y))
    cate_xgb = xgbt.fit_predict(X, treatment, y).ravel()

    ate_results = pd.DataFrame(
        [
            {
                "Model": "LRSRegressor",
                "ATE_log1p": ate_lr[0],
                "ATE_CI_lower": ate_lr[1],
                "ATE_CI_upper": ate_lr[2],
            },
            {
                "Model": "XGBTRegressor",
                "ATE_log1p": ate_xgb[0],
                "ATE_CI_lower": ate_xgb[1],
                "ATE_CI_upper": ate_xgb[2],
            },
        ]
    )

    uplift_predictions = work[["Customer ID", outcome_col, "TreatmentGroup", "PropensityScore"]].copy()
    uplift_predictions["CATE_LRSRegressor"] = cate_lr
    uplift_predictions["CATE_XGBTRegressor"] = cate_xgb

    lr_model = lrs.models["campaign"]
    lr_coef = np.asarray(lr_model.coefficients, dtype=float)
    # StatsmodelsOLS stores [intercept, features..., treatment]
    lr_feature_coef = np.abs(lr_coef[1 : 1 + len(feature_cols)])

    xgb_model_control = xgbt.models_c["campaign"]
    xgb_model_treated = xgbt.models_t["campaign"]
    xgb_importance = (
        np.asarray(xgb_model_control.feature_importances_, dtype=float)
        + np.asarray(xgb_model_treated.feature_importances_, dtype=float)
    ) / 2.0

    feature_importance = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance_LRSRegressor": lr_feature_coef,
            "Importance_XGBTRegressor": xgb_importance,
        }
    )
    feature_importance["Importance_Avg"] = feature_importance[
        ["Importance_LRSRegressor", "Importance_XGBTRegressor"]
    ].mean(axis=1)
    feature_importance = feature_importance.sort_values("Importance_Avg", ascending=False)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ate_results.to_csv(output_dir / "causal_ate_results.csv", index=False)
        uplift_predictions.to_csv(output_dir / "causal_uplift_predictions.csv", index=False)
        feature_importance.to_csv(output_dir / "causal_feature_importance.csv", index=False)

        plt.figure(figsize=(9, 5))
        sns.histplot(cate_lr, kde=True, color="steelblue", label="LRSRegressor", stat="density")
        sns.histplot(cate_xgb, kde=True, color="darkorange", label="XGBTRegressor", stat="density")
        plt.axvline(np.mean(cate_lr), color="steelblue", linestyle="--", linewidth=1.5)
        plt.axvline(np.mean(cate_xgb), color="darkorange", linestyle="--", linewidth=1.5)
        plt.title("Estimated Individual Treatment Effects (CATE)")
        plt.xlabel("Estimated uplift on log1p(CLV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "causal_cate_distribution.png", dpi=150)
        plt.close()

        top_features = feature_importance.head(12).sort_values("Importance_Avg")
        plt.figure(figsize=(9, 6))
        plt.barh(top_features["Feature"], top_features["Importance_Avg"], color="teal")
        plt.title("Causal Feature Importance (Average)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(output_dir / "causal_feature_importance.png", dpi=150)
        plt.close()

    return CausalRunOutput(
        ate_results=ate_results,
        uplift_predictions=uplift_predictions,
        feature_importance=feature_importance,
    )
