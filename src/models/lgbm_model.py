"""LightGBM model wrapper for time series forecasting.

Optimized for large datasets (4M+ records) with focus on training speed
and memory efficiency while maintaining accuracy.

XGBoost to LightGBM parameter mapping reference:
    XGBoost              -> LightGBM
    n_estimators         -> n_estimators (same)
    max_depth            -> max_depth (same, but use -1 for no limit)
    learning_rate        -> learning_rate (same)
    subsample            -> subsample / bagging_fraction
    colsample_bytree     -> colsample_bytree / feature_fraction
    min_child_weight     -> min_child_samples (concept differs)
    gamma                -> min_split_gain
    reg_alpha            -> reg_alpha (L1)
    reg_lambda           -> reg_lambda (L2)
    tree_method='hist'   -> (default in LightGBM)
    n_jobs               -> n_jobs
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not installed. Install with: pip install lightgbm")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Install with: pip install optuna")

from sklearn.model_selection import TimeSeriesSplit


@dataclass
class LGBMConfig:
    """Configuration for LightGBM model."""

    # Core parameters
    objective: str = "regression"
    metric: str = "rmse"
    boosting_type: str = "gbdt"

    # Tree parameters
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20

    # Learning parameters
    learning_rate: float = 0.05
    n_estimators: int = 1000

    # Regularization
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    min_split_gain: float = 0.0

    # Sampling
    subsample: float = 1.0
    subsample_freq: int = 0
    colsample_bytree: float = 1.0

    # Large dataset optimizations
    max_bin: int = 255
    min_data_in_bin: int = 3

    # Other
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1
    device: str = "cpu"

    # Early stopping
    early_stopping_rounds: int = 50

    @classmethod
    def from_xgboost_params(cls, xgb_params: Dict[str, Any]) -> "LGBMConfig":
        """Create LGBMConfig from XGBoost parameters.

        Automatically maps XGBoost hyperparameters to LightGBM equivalents.

        Args:
            xgb_params: Dictionary of XGBoost parameters

        Returns:
            LGBMConfig with mapped parameters
        """
        lgb_params = {}

        # Direct mappings (same name)
        direct_map = ['learning_rate', 'reg_alpha', 'reg_lambda',
                      'random_state', 'n_jobs']
        for param in direct_map:
            if param in xgb_params:
                lgb_params[param] = xgb_params[param]

        # n_estimators mapping
        if 'n_estimators' in xgb_params:
            lgb_params['n_estimators'] = xgb_params['n_estimators']

        # max_depth mapping (XGBoost default 6, LightGBM uses -1 for unlimited)
        if 'max_depth' in xgb_params:
            lgb_params['max_depth'] = xgb_params['max_depth']

        # subsample -> subsample (same name in sklearn API)
        if 'subsample' in xgb_params:
            lgb_params['subsample'] = xgb_params['subsample']
            lgb_params['subsample_freq'] = 1

        # colsample_bytree -> colsample_bytree (same name in sklearn API)
        if 'colsample_bytree' in xgb_params:
            lgb_params['colsample_bytree'] = xgb_params['colsample_bytree']

        # gamma -> min_split_gain
        if 'gamma' in xgb_params:
            lgb_params['min_split_gain'] = xgb_params['gamma']

        # min_child_weight -> min_child_samples (approximate conversion)
        if 'min_child_weight' in xgb_params:
            lgb_params['min_child_samples'] = int(xgb_params['min_child_weight'] * 15)

        # Set num_leaves based on max_depth (LightGBM best practice)
        if 'max_depth' in lgb_params:
            lgb_params['num_leaves'] = min(2 ** lgb_params['max_depth'] - 1, 127)

        return cls(**lgb_params)

    @classmethod
    def for_large_dataset(cls, n_records: int) -> "LGBMConfig":
        """Get optimized config for large datasets.

        Args:
            n_records: Number of records in dataset

        Returns:
            LGBMConfig optimized for the dataset size
        """
        if n_records < 100_000:
            return cls(
                num_leaves=31,
                min_child_samples=20,
                max_bin=255,
            )
        elif n_records < 1_000_000:
            return cls(
                num_leaves=63,
                min_child_samples=50,
                max_bin=255,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
            )
        elif n_records < 10_000_000:
            # 4M records falls here - optimized for speed and memory
            return cls(
                num_leaves=127,
                min_child_samples=100,
                max_bin=511,  # More bins for better accuracy
                min_data_in_bin=5,
                subsample=0.7,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_alpha=0.2,
                reg_lambda=2.0,
                learning_rate=0.05,
                n_estimators=500,
            )
        else:
            # 10M+ records
            return cls(
                num_leaves=255,
                min_child_samples=200,
                max_bin=1023,
                min_data_in_bin=10,
                subsample=0.6,
                subsample_freq=1,
                colsample_bytree=0.6,
                reg_alpha=0.5,
                reg_lambda=5.0,
                learning_rate=0.03,
                n_estimators=1000,
            )

    def to_dict(self) -> dict:
        """Convert to dictionary for LightGBM."""
        return {
            "objective": self.objective,
            "metric": self.metric,
            "boosting_type": self.boosting_type,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_split_gain": self.min_split_gain,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "max_bin": self.max_bin,
            "min_data_in_bin": self.min_data_in_bin,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "device": self.device,
        }


class LGBMModel:
    """LightGBM model wrapper for traffic prediction.

    Gradient boosting model optimized for tabular data with
    support for Optuna hyperparameter tuning.
    """

    def __init__(self, config: LGBMConfig | None = None):
        """Initialize LightGBM model.

        Args:
            config: Model configuration
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

        self.config = config or LGBMConfig()
        self.model = None
        self.feature_names = None
        self.best_iteration = None

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        feature_names: list[str] | None = None,
        verbose: bool = True,
    ) -> "LGBMModel":
        """Fit LightGBM model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target
            feature_names: Feature names
            verbose: Whether to print progress

        Returns:
            Self for method chaining
        """
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        elif feature_names is not None:
            self.feature_names = feature_names

        # Create model
        params = self.config.to_dict()
        self.model = lgb.LGBMRegressor(**params)

        # Prepare callbacks
        callbacks = []
        if X_val is not None and self.config.early_stopping_rounds:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds)
            )
        if verbose:
            callbacks.append(lgb.log_evaluation(period=100))

        # Fit model
        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )

        self.best_iteration = self.model.best_iteration_

        if verbose and self.best_iteration:
            print(f"Best iteration: {self.best_iteration}")

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Features

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)

    def predict_with_interval(
        self,
        X: pd.DataFrame | np.ndarray,
        alpha: float = 0.05,
        n_iterations: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with prediction intervals.

        Uses bootstrap resampling to estimate intervals.

        Args:
            X: Features
            alpha: Significance level
            n_iterations: Number of bootstrap iterations

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        predictions = self.predict(X)

        # Estimate prediction intervals using residual bootstrap
        # This is a simple approximation
        residuals = getattr(self, "_residuals", None)
        if residuals is None:
            # If no residuals stored, use a simple heuristic
            std_estimate = np.std(predictions) * 0.1
            lower = predictions - 1.96 * std_estimate
            upper = predictions + 1.96 * std_estimate
        else:
            # Bootstrap residuals
            bootstrap_preds = []
            for _ in range(n_iterations):
                sampled_residuals = np.random.choice(residuals, size=len(predictions))
                bootstrap_preds.append(predictions + sampled_residuals)

            bootstrap_preds = np.array(bootstrap_preds)
            lower = np.percentile(bootstrap_preds, 100 * alpha / 2, axis=0)
            upper = np.percentile(bootstrap_preds, 100 * (1 - alpha / 2), axis=0)

        return predictions, lower, upper

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """Get feature importance.

        Args:
            importance_type: Type of importance ('gain', 'split', 'weight')

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        importance = self.model.feature_importances_

        return pd.DataFrame({
            "feature": self.feature_names or [f"f{i}" for i in range(len(importance))],
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def get_shap_values(
        self,
        X: pd.DataFrame | np.ndarray,
        max_samples: int | None = 1000,
    ) -> np.ndarray:
        """Get SHAP values for feature importance.

        Args:
            X: Features
            max_samples: Maximum samples for SHAP calculation

        Returns:
            SHAP values array
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")

        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Sample if too large
        if max_samples and len(X) > max_samples:
            if isinstance(X, pd.DataFrame):
                X = X.sample(n=max_samples, random_state=42)
            else:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X = X[indices]

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        return shap_values

    def tune_hyperparameters(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        n_trials: int = 100,
        n_splits: int = 5,
        gap: int = 12,
        verbose: bool = True,
    ) -> dict:
        """Tune hyperparameters using Optuna.

        Args:
            X: Features
            y: Target
            n_trials: Number of Optuna trials
            n_splits: Number of CV splits
            gap: Gap between train and validation in TimeSeriesSplit
            verbose: Whether to print progress

        Returns:
            Best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Install with: pip install optuna")

        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                # Stronger regularization: smaller num_leaves, shallower depth
                "num_leaves": trial.suggest_int("num_leaves", 10, 50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "n_estimators": 1000,
                # Fixed: Use LINEAR scale for regularization (not log) with meaningful range
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0),  # Linear scale
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),  # Linear scale
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                if isinstance(X, pd.DataFrame):
                    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                else:
                    X_train_cv, X_val_cv = X[train_idx], X[val_idx]

                if isinstance(y, pd.Series):
                    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                else:
                    y_train_cv, y_val_cv = y[train_idx], y[val_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train_cv,
                    y_train_cv,
                    eval_set=[(X_val_cv, y_val_cv)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )

                preds = model.predict(X_val_cv)
                rmse = np.sqrt(np.mean((y_val_cv - preds) ** 2))
                scores.append(rmse)

            return np.mean(scores)

        # Create study
        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=verbose,
        )

        if verbose:
            print(f"\nBest trial: {study.best_trial.number}")
            print(f"Best RMSE: {study.best_trial.value:.4f}")
            print(f"Best params: {study.best_trial.params}")

        return study.best_trial.params

    def cross_validate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        n_splits: int = 5,
        gap: int = 12,
    ) -> dict:
        """Perform time series cross-validation.

        Args:
            X: Features
            y: Target
            n_splits: Number of CV splits
            gap: Gap between train and validation

        Returns:
            Dictionary with CV scores
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        scores = {"rmse": [], "mae": [], "mape": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]

            if isinstance(y, pd.Series):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]

            # Create new model for each fold
            model = lgb.LGBMRegressor(**self.config.to_dict())
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )

            preds = model.predict(X_val)

            # Calculate metrics
            rmse = np.sqrt(np.mean((y_val - preds) ** 2))
            mae = np.mean(np.abs(y_val - preds))
            mape = np.mean(np.abs((y_val - preds) / (y_val + 1e-8))) * 100

            scores["rmse"].append(rmse)
            scores["mae"].append(mae)
            scores["mape"].append(mape)

        # Add summary statistics (iterate over a list copy to avoid modification during iteration)
        for metric in list(scores.keys()):
            scores[f"{metric}_mean"] = np.mean(scores[metric])
            scores[f"{metric}_std"] = np.std(scores[metric])

        return scores

    def save(self, filepath: str | Path) -> None:
        """Save model to file.

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "config": self.config,
            "model": self.model,
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration,
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "LGBMModel":
        """Load model from file.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded model
        """
        model_data = joblib.load(filepath)
        instance = cls(config=model_data["config"])
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.best_iteration = model_data["best_iteration"]
        return instance

    @classmethod
    def from_xgboost_params(cls, xgb_params: Dict[str, Any]) -> "LGBMModel":
        """Create LGBMModel from XGBoost parameters.

        Convenience method that maps XGBoost hyperparameters to LightGBM.

        Args:
            xgb_params: Dictionary of XGBoost parameters

        Returns:
            LGBMModel with mapped parameters

        Example:
            >>> xgb_params = {
            ...     'n_estimators': 500,
            ...     'max_depth': 6,
            ...     'learning_rate': 0.05,
            ...     'subsample': 0.8,
            ...     'colsample_bytree': 0.8,
            ...     'min_child_weight': 3,
            ...     'gamma': 0.1,
            ...     'reg_alpha': 0.1,
            ...     'reg_lambda': 1.0,
            ... }
            >>> model = LGBMModel.from_xgboost_params(xgb_params)
        """
        config = LGBMConfig.from_xgboost_params(xgb_params)
        return cls(config=config)


def get_gpu_config() -> Dict[str, Any]:
    """Get parameters for GPU training.

    Returns:
        Dictionary of GPU-specific parameters to add to config
    """
    return {
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "gpu_use_dp": False,  # Use single precision for speed
    }


def migrate_xgboost_to_lightgbm(xgb_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert XGBoost parameters to LightGBM format.

    This function provides a detailed mapping of all common XGBoost
    parameters to their LightGBM equivalents.

    Args:
        xgb_params: Dictionary of XGBoost parameters

    Returns:
        Dictionary of LightGBM parameters

    Parameter Mapping Reference:
        XGBoost                    -> LightGBM
        -------                       --------
        n_estimators               -> n_estimators
        max_depth                  -> max_depth
        learning_rate (eta)        -> learning_rate
        subsample                  -> bagging_fraction / subsample
        colsample_bytree           -> feature_fraction / colsample_bytree
        colsample_bylevel          -> (not directly supported)
        colsample_bynode           -> feature_fraction_bynode
        min_child_weight           -> min_child_samples (different concept)
        gamma (min_split_loss)     -> min_split_gain
        reg_alpha (alpha)          -> reg_alpha (lambda_l1)
        reg_lambda (lambda)        -> reg_lambda (lambda_l2)
        max_delta_step             -> (not directly supported)
        scale_pos_weight           -> scale_pos_weight
        tree_method='hist'         -> (default in LightGBM)
        grow_policy='lossguide'    -> boosting_type='gbdt' (default)
        n_jobs                     -> n_jobs / num_threads
        random_state               -> random_state / seed
        verbosity                  -> verbose

    Notes:
        - LightGBM uses leaf-wise growth by default (more efficient)
        - min_child_weight in XGBoost relates to sum of instance weight (hessian)
        - min_child_samples in LightGBM is a count of samples
        - For similar behavior, multiply min_child_weight by ~10-20
    """
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbose": -1,
    }

    # Direct mappings
    direct_map = {
        'n_estimators': 'n_estimators',
        'learning_rate': 'learning_rate',
        'reg_alpha': 'reg_alpha',
        'reg_lambda': 'reg_lambda',
        'random_state': 'random_state',
        'n_jobs': 'n_jobs',
        'scale_pos_weight': 'scale_pos_weight',
    }

    for xgb_name, lgb_name in direct_map.items():
        if xgb_name in xgb_params:
            lgb_params[lgb_name] = xgb_params[xgb_name]

    # max_depth mapping
    if 'max_depth' in xgb_params:
        lgb_params['max_depth'] = xgb_params['max_depth']
        # Set num_leaves based on max_depth (LightGBM recommendation)
        lgb_params['num_leaves'] = min(2 ** xgb_params['max_depth'] - 1, 127)

    # Sampling parameters
    if 'subsample' in xgb_params:
        lgb_params['bagging_fraction'] = xgb_params['subsample']
        lgb_params['bagging_freq'] = 1  # Enable bagging

    if 'colsample_bytree' in xgb_params:
        lgb_params['feature_fraction'] = xgb_params['colsample_bytree']

    if 'colsample_bynode' in xgb_params:
        lgb_params['feature_fraction_bynode'] = xgb_params['colsample_bynode']

    # gamma -> min_split_gain
    if 'gamma' in xgb_params:
        lgb_params['min_split_gain'] = xgb_params['gamma']

    # min_child_weight -> min_child_samples (approximate conversion)
    if 'min_child_weight' in xgb_params:
        # XGBoost: sum of instance weight (hessian)
        # LightGBM: count of samples
        # Approximate conversion factor: 10-20
        lgb_params['min_child_samples'] = int(xgb_params['min_child_weight'] * 15)

    return lgb_params
