#!/usr/bin/env python3
"""Run the end-to-end CLV project pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from causal import CAUSALML_AVAILABLE, create_campaign_treatment, run_causal_analysis
from data_loader import clean_data, ensure_raw_data, load_raw_data
from features import calculate_clv_target, create_customer_features, split_data_temporal
from models import (
    cross_validate_model,
    evaluate_model,
    get_feature_importance,
    get_models,
    get_segment_labels,
    segment_customers_by_clv,
    train_and_evaluate,
    tune_model,
)
from monitoring import create_monitoring_artifacts

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    SHAP_AVAILABLE = False


RANDOM_STATE = 42
FEATURE_COLS = [
    "Recency",
    "Frequency",
    "Monetary",
    "Tenure",
    "AvgTimeBetweenPurchases",
    "NumUniqueProducts",
    "AvgBasketSize",
    "AvgOrderValue",
]


def _ensure_dirs() -> Dict[str, Path]:
    processed_dir = ROOT / "data" / "processed"
    figures_dir = ROOT / "reports" / "figures"
    reports_dir = ROOT / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return {
        "processed_dir": processed_dir,
        "figures_dir": figures_dir,
        "reports_dir": reports_dir,
    }


def _save_model_comparison(results_df: pd.DataFrame, figures_dir: Path) -> None:
    metrics = ["RMSE", "MAE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, metrics):
        order = results_df.sort_values(metric, ascending=(metric != "R2"))
        ax.bar(order.index, order[metric], color="#2f6690", edgecolor="black")
        ax.set_title(f"{metric} by Model")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison.png", dpi=160)
    plt.close()


def _save_pred_vs_actual_plot(y_test: pd.Series, y_pred: np.ndarray, figures_dir: Path, model_name: str) -> None:
    y_pred = np.clip(y_pred, 0, None)
    residuals = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test.values, y_pred, alpha=0.45, s=12)
    diagonal_max = max(float(y_test.max()), float(np.max(y_pred)))
    axes[0].plot([0, diagonal_max], [0, diagonal_max], "r--", linewidth=1.5)
    axes[0].set_title(f"{model_name}: Predicted vs Actual")
    axes[0].set_xlabel("Actual CLV")
    axes[0].set_ylabel("Predicted CLV")

    axes[1].hist(residuals, bins=50, color="#3f7d20", edgecolor="black", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title(f"{model_name}: Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "best_model_analysis.png", dpi=160)
    plt.close()


def _save_lift_plot(y_true: np.ndarray, y_pred: np.ndarray, figures_dir: Path) -> Dict[str, float]:
    frame = pd.DataFrame({"Actual_CLV": y_true, "Predicted_CLV": y_pred})
    frame = frame.sort_values("Predicted_CLV", ascending=False).reset_index(drop=True)
    frame["CustomerPct"] = (np.arange(len(frame)) + 1) / len(frame)
    frame["CumActualPct"] = frame["Actual_CLV"].cumsum() / max(frame["Actual_CLV"].sum(), 1)

    plt.figure(figsize=(10, 6))
    plt.plot(frame["CustomerPct"] * 100, frame["CumActualPct"] * 100, label="Model", linewidth=2)
    plt.plot([0, 100], [0, 100], "r--", label="Random", linewidth=1.8)
    plt.xlabel("% Customers Contacted (sorted by predicted CLV)")
    plt.ylabel("% Realized CLV Captured")
    plt.title("Lift Chart")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "lift_chart.png", dpi=160)
    plt.close()

    lift_stats: Dict[str, float] = {}
    for pct in (10, 20, 30, 50):
        top_n = max(1, int(len(frame) * pct / 100))
        captured = frame.head(top_n)["Actual_CLV"].sum() / max(frame["Actual_CLV"].sum(), 1)
        lift = captured / (pct / 100)
        lift_stats[f"Lift@{pct}%"] = float(lift)
    return lift_stats


def _try_save_shap_plot(
    model,
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    figures_dir: Path,
) -> bool:
    if not SHAP_AVAILABLE:
        return False

    try:
        sample_train = X_train_scaled.sample(min(1000, len(X_train_scaled)), random_state=RANDOM_STATE)
        sample_test = X_test_scaled.sample(min(500, len(X_test_scaled)), random_state=RANDOM_STATE)

        explainer = shap.Explainer(model, sample_train)
        shap_values = explainer(sample_test)
        plt.figure()
        shap.summary_plot(shap_values, sample_test, show=False)
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        return False


def _save_feature_importance(
    model_name: str,
    model,
    feature_cols: List[str],
    reports_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    fi = get_feature_importance(model, feature_cols, model_name=model_name)
    if fi.empty:
        fi = pd.DataFrame({"Feature": feature_cols, "Importance": 0.0})
    fi.to_csv(reports_dir / "feature_importance.csv", index=False)

    top_fi = fi.sort_values("Importance", ascending=True).tail(12)
    plt.figure(figsize=(9, 6))
    plt.barh(top_fi["Feature"], top_fi["Importance"], color="#005f73")
    plt.title(f"Feature Importance ({model_name})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=160)
    plt.close()
    return fi


def run_pipeline(force_download: bool = False, skip_causal: bool = False) -> None:
    dirs = _ensure_dirs()
    processed_dir = dirs["processed_dir"]
    figures_dir = dirs["figures_dir"]
    reports_dir = dirs["reports_dir"]

    raw_path = ensure_raw_data(force_download=force_download)
    raw_df = load_raw_data(raw_path, download_if_missing=False)
    clean_df = clean_data(raw_df)
    clean_df.to_csv(processed_dir / "cleaned_retail.csv", index=False)

    obs_df, pred_df, obs_end, pred_end = split_data_temporal(
        clean_df,
        date_col="InvoiceDate",
        observation_months=12,
        prediction_months=6,
    )
    features_df = create_customer_features(
        obs_df,
        customer_id_col="Customer ID",
        date_col="InvoiceDate",
        amount_col="TotalAmount",
        invoice_col="Invoice",
        quantity_col="Quantity",
        product_col="StockCode",
        reference_date=obs_end,
    )
    clv_df = calculate_clv_target(
        observation_df=obs_df,
        prediction_df=pred_df,
        customer_id_col="Customer ID",
        amount_col="TotalAmount",
    )
    dataset = features_df.merge(clv_df, on="Customer ID", how="left").fillna(0)
    dataset.to_csv(processed_dir / "customer_features.csv", index=False)

    X = dataset[FEATURE_COLS].copy().fillna(0)
    y = dataset["CLV"].astype(float).fillna(0)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )
    print(f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    base_models = get_models(include_xgboost=True)
    base_results, trained_models, scaler = train_and_evaluate(
        X_train=X_train,
        X_test=X_val,
        y_train=y_train,
        y_test=y_val,
        models=base_models,
        scale_features=True,
    )
    baseline_df = pd.DataFrame(base_results).T

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=FEATURE_COLS, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=FEATURE_COLS, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
    )

    # Fine-tune top candidate models by baseline RMSE.
    tuned_records = []
    ranked = baseline_df.sort_values("RMSE").index.tolist()[:3]
    for model_name in ranked:
        tuned_model, best_params, best_cv_rmse = tune_model(
            model_name=model_name,
            model=base_models[model_name],
            X_train=X_train_scaled,
            y_train=y_train,
            cv=5,
            n_iter=18,
            random_state=RANDOM_STATE,
        )
        if not best_params:
            continue
        tuned_model.fit(X_train_scaled, y_train)
        tuned_pred = np.clip(tuned_model.predict(X_val_scaled), 0, None)
        tuned_metrics = evaluate_model(y_val.values, tuned_pred)
        tuned_name = f"{model_name} (Tuned)"
        tuned_records.append(
            {
                "Model": tuned_name,
                **tuned_metrics,
                "BestParams": json.dumps(best_params),
                "CV_RMSE": best_cv_rmse,
            }
        )
        trained_models[tuned_name] = tuned_model

    tuned_df = pd.DataFrame(tuned_records).set_index("Model") if tuned_records else pd.DataFrame()
    results_df = baseline_df.copy()
    if not tuned_df.empty:
        results_df = pd.concat([results_df, tuned_df[["RMSE", "MAE", "R2", "MAPE"]]], axis=0)

    results_df = results_df.sort_values("RMSE")
    results_df.to_csv(reports_dir / "model_results.csv")
    _save_model_comparison(results_df, figures_dir)

    cv_rows = []
    for name in results_df.index.tolist():
        model = trained_models[name] if name in trained_models else base_models[name]
        cv_metrics = cross_validate_model(
            model=model,
            X=X_train,
            y=y_train,
            cv=5,
            scale_features=True,
        )
        cv_rows.append({"Model": name, **cv_metrics})
    cv_df = pd.DataFrame(cv_rows).sort_values("RMSE_mean")
    cv_df.to_csv(reports_dir / "cv_results.csv", index=False)

    best_model_name = results_df.index[0]
    best_model = trained_models[best_model_name]
    y_pred_best = np.clip(best_model.predict(X_test_scaled), 0, None)

    _save_pred_vs_actual_plot(y_test, y_pred_best, figures_dir, best_model_name)
    lift_stats = _save_lift_plot(y_test.values, y_pred_best, figures_dir)
    fi = _save_feature_importance(best_model_name, best_model, FEATURE_COLS, reports_dir, figures_dir)
    shap_saved = _try_save_shap_plot(best_model, X_train_scaled, X_test_scaled, figures_dir)

    X_all_scaled = pd.DataFrame(
        scaler.transform(X), columns=FEATURE_COLS, index=X.index
    )
    all_predictions = np.clip(best_model.predict(X_all_scaled), 0, None)

    pred_df = dataset[["Customer ID", "CLV"]].copy()
    pred_df["Predicted_CLV"] = all_predictions
    pred_df.to_csv(processed_dir / "clv_predictions.csv", index=False)

    segments = segment_customers_by_clv(all_predictions, n_segments=4)
    segment_labels = get_segment_labels(n_segments=4)
    seg_df = pred_df.copy()
    seg_df["Segment"] = segments
    seg_df["SegmentLabel"] = seg_df["Segment"].map(segment_labels)
    seg_df.to_csv(processed_dir / "customer_segments.csv", index=False)

    create_monitoring_artifacts(
        baseline_df=X_train,
        current_df=X_test,
        output_dir=reports_dir,
        model_name=best_model_name,
    )

    causal_executed = False
    if not skip_causal and CAUSALML_AVAILABLE:
        causal_dataset = create_campaign_treatment(dataset, random_state=RANDOM_STATE)
        causal_dataset.to_csv(processed_dir / "causal_dataset.csv", index=False)
        run_causal_analysis(
            df=causal_dataset,
            feature_cols=FEATURE_COLS,
            outcome_col="CLV",
            output_dir=reports_dir,
            control_name="control",
            random_state=RANDOM_STATE,
        )
        causal_executed = True

    metadata = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "observation_period_end": str(obs_end.date()),
        "prediction_period_end": str(pred_end.date()),
        "num_raw_rows": int(len(raw_df)),
        "num_clean_rows": int(len(clean_df)),
        "num_customers": int(dataset["Customer ID"].nunique()),
        "best_model": best_model_name,
        "shap_summary_generated": shap_saved,
        "causal_analysis_executed": causal_executed,
        "lift_stats": lift_stats,
        "top_features": fi.head(8).to_dict(orient="records"),
    }
    (reports_dir / "pipeline_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if tuned_records:
        pd.DataFrame(tuned_records).to_csv(reports_dir / "tuning_results.csv", index=False)

    print("Pipeline completed successfully.")
    print(f"Best model: {best_model_name}")
    print(f"SHAP generated: {shap_saved}")
    print(f"Causal phase executed: {causal_executed}")
    print(f"Artifacts saved under: {processed_dir} and {reports_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full CLV project pipeline.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of the raw dataset from UCI.",
    )
    parser.add_argument(
        "--skip-causal",
        action="store_true",
        help="Skip CausalML phase.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(force_download=args.force_download, skip_causal=args.skip_causal)
