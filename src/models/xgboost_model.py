"""XGBoost model for traffic prediction."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class XGBoostModel:
    """XGBoost regressor for time series forecasting.
    
    Optimized for NASA web traffic prediction with hyperparameters
    tuned for handling outliers and temporal patterns.
    """

    def __init__(self, **params):
        """Initialize XGBoost model.

        Args:
            **params: XGBoost parameters. If not provided, uses optimized defaults.
        """
        # Default parameters optimized for time series with outliers
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'tree_method': 'hist',  # Faster for large datasets
        }
        default_params.update(params)
        
        self.model = xgb.XGBRegressor(**default_params)
        self.params = default_params
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """Train XGBoost model.

        Args:
            X: Training features (DataFrame or array)
            y: Target values
            eval_set: Optional validation set for early stopping
            early_stopping_rounds: Stop if no improvement after N rounds
            verbose: Print training progress

        Returns:
            Self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        
        fit_params = {'verbose': verbose}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        if early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds

        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Make predictions.

        Args:
            X: Features (DataFrame or array)

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate model performance.

        Args:
            X: Features
            y: True values

        Returns:
            Dictionary with metrics: rmse, mae, mape, r2
        """
        y_pred = self.predict(X)
        
        # Handle edge cases for MAPE
        mask = y != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
        else:
            mape = np.nan

        return {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'mape': mape,
            'r2': r2_score(y, y_pred),
        }

    def get_feature_importance(self, importance_type='weight'):
        """Get feature importance scores.

        Args:
            importance_type: 'weight', 'gain', or 'cover'

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")

        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        if self.feature_names is not None:
            # Map feature indices to names
            importance_df = pd.DataFrame([
                {'feature': self.feature_names[int(k.replace('f', ''))], 'importance': v}
                for k, v in importance.items()
            ])
        else:
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v}
                for k, v in importance.items()
            ])

        return importance_df.sort_values('importance', ascending=False)

    def save(self, filepath):
        """Save model to file.

        Args:
            filepath: Path to save model (JSON format)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model.")
        
        self.model.save_model(filepath)

    def load(self, filepath):
        """Load model from file.

        Args:
            filepath: Path to model file
        """
        self.model.load_model(filepath)
        self.is_fitted = True
        return self
