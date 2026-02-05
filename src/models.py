"""
Model training utilities for CLV prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings


def get_models() -> Dict[str, Any]:
    """
    Get dictionary of models to train.

    Returns:
        Dictionary mapping model names to model instances.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    return models


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model predictions.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary with evaluation metrics.
    """
    # Clip negative predictions to 0 (CLV can't be negative)
    y_pred = np.clip(y_pred, 0, None)

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100  # +1 to avoid div by 0
    }
    return metrics


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: Dict[str, Any] = None,
    scale_features: bool = True
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any], StandardScaler]:
    """
    Train multiple models and evaluate them.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        models: Dictionary of models to train. If None, uses default models.
        scale_features: Whether to scale features.

    Returns:
        Tuple of (results dict, trained models dict, scaler)
    """
    if models is None:
        models = get_models()

    scaler = None
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

    results = {}
    trained_models = {}

    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Train model
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Evaluate
            results[name] = evaluate_model(y_test.values, y_pred)

    return results, trained_models, scaler


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """
    Get feature importance from a trained model.

    Args:
        model: Trained model.
        feature_names: List of feature names.
        model_name: Name of the model.

    Returns:
        DataFrame with feature importances.
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
    else:
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    return importance_df


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scale_features: bool = True
) -> Dict[str, float]:
    """
    Cross-validate a model.

    Args:
        model: Model to cross-validate.
        X: Features.
        y: Target.
        cv: Number of folds.
        scale_features: Whether to scale features.

    Returns:
        Dictionary with cross-validation metrics.
    """
    X_scaled = X.copy()

    if scale_features:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

    # Cross-validation scores
    r2_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")
    neg_rmse_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="neg_root_mean_squared_error")
    neg_mae_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="neg_mean_absolute_error")

    return {
        "R2_mean": r2_scores.mean(),
        "R2_std": r2_scores.std(),
        "RMSE_mean": -neg_rmse_scores.mean(),
        "RMSE_std": neg_rmse_scores.std(),
        "MAE_mean": -neg_mae_scores.mean(),
        "MAE_std": neg_mae_scores.std()
    }


def segment_customers_by_clv(
    predictions: np.ndarray,
    n_segments: int = 4
) -> np.ndarray:
    """
    Segment customers based on predicted CLV using quantiles.

    Args:
        predictions: Predicted CLV values.
        n_segments: Number of segments.

    Returns:
        Array of segment labels.
    """
    quantiles = np.linspace(0, 100, n_segments + 1)
    thresholds = np.percentile(predictions, quantiles)

    segments = np.digitize(predictions, thresholds[1:-1])

    return segments


def get_segment_labels(n_segments: int = 4) -> Dict[int, str]:
    """
    Get human-readable segment labels.

    Args:
        n_segments: Number of segments.

    Returns:
        Dictionary mapping segment numbers to labels.
    """
    if n_segments == 4:
        return {
            0: "Low Value",
            1: "Medium-Low Value",
            2: "Medium-High Value",
            3: "High Value (Champions)"
        }
    elif n_segments == 3:
        return {
            0: "Low Value",
            1: "Medium Value",
            2: "High Value"
        }
    else:
        return {i: f"Segment {i+1}" for i in range(n_segments)}
