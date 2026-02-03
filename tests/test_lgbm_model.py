"""Tests for LightGBM model wrapper."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.models.lgbm_model import (
    LGBMConfig,
    LGBMModel,
    get_gpu_config,
    migrate_xgboost_to_lightgbm,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_train_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 200

    X = pd.DataFrame({
        "feature_0": np.random.randn(n_samples),
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "lag_1": np.random.randn(n_samples),
        "lag_2": np.random.randn(n_samples),
    })

    # Target with relationship to features
    y = 50 + 10 * X["feature_0"] + 5 * X["feature_1"] + np.random.randn(n_samples) * 3

    return X, pd.Series(y, name="target")


@pytest.fixture
def sample_train_val_data(sample_train_data):
    """Create train/validation split."""
    X, y = sample_train_data
    split_idx = int(len(X) * 0.8)

    return {
        "X_train": X.iloc[:split_idx],
        "y_train": y.iloc[:split_idx],
        "X_val": X.iloc[split_idx:],
        "y_val": y.iloc[split_idx:],
    }


@pytest.fixture
def fitted_model(sample_train_data):
    """Create a fitted LightGBM model."""
    X, y = sample_train_data
    model = LGBMModel()
    model.fit(X, y, verbose=False)
    return model


@pytest.fixture
def tmp_model_path(tmp_path):
    """Create temporary path for model saving."""
    return tmp_path / "test_model.pkl"


# =============================================================================
# LGBMConfig Tests
# =============================================================================


class TestLGBMConfig:
    """Tests for LGBMConfig dataclass."""

    def test_default_initialization(self):
        """Test default config values."""
        config = LGBMConfig()

        assert config.objective == "regression"
        assert config.metric == "rmse"
        assert config.boosting_type == "gbdt"
        assert config.num_leaves == 31
        assert config.max_depth == -1
        assert config.learning_rate == 0.05
        assert config.n_estimators == 1000
        assert config.random_state == 42

    def test_custom_initialization(self):
        """Test config with custom values."""
        config = LGBMConfig(
            num_leaves=63,
            max_depth=8,
            learning_rate=0.1,
            n_estimators=500,
        )

        assert config.num_leaves == 63
        assert config.max_depth == 8
        assert config.learning_rate == 0.1
        assert config.n_estimators == 500

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LGBMConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "objective" in config_dict
        assert "num_leaves" in config_dict
        assert "learning_rate" in config_dict
        assert config_dict["objective"] == "regression"

    def test_from_xgboost_params_basic(self):
        """Test XGBoost parameter mapping."""
        xgb_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
        }

        config = LGBMConfig.from_xgboost_params(xgb_params)

        assert config.n_estimators == 500
        assert config.max_depth == 6
        assert config.learning_rate == 0.1

    def test_from_xgboost_params_gamma_mapping(self):
        """Test gamma -> min_split_gain mapping."""
        xgb_params = {"gamma": 0.5}
        config = LGBMConfig.from_xgboost_params(xgb_params)

        assert config.min_split_gain == 0.5

    def test_from_xgboost_params_min_child_weight_mapping(self):
        """Test min_child_weight -> min_child_samples mapping."""
        xgb_params = {"min_child_weight": 3}
        config = LGBMConfig.from_xgboost_params(xgb_params)

        # Conversion factor is ~15
        assert config.min_child_samples == 45

    def test_from_xgboost_params_subsample(self):
        """Test subsample parameter mapping."""
        xgb_params = {"subsample": 0.8}
        config = LGBMConfig.from_xgboost_params(xgb_params)

        assert config.subsample == 0.8
        assert config.subsample_freq == 1  # Enabled when subsample < 1

    def test_from_xgboost_params_num_leaves_calculation(self):
        """Test num_leaves is calculated from max_depth."""
        xgb_params = {"max_depth": 5}
        config = LGBMConfig.from_xgboost_params(xgb_params)

        # num_leaves = min(2^max_depth - 1, 127) = min(31, 127) = 31
        assert config.num_leaves == 31

    def test_for_large_dataset_small(self):
        """Test config for small dataset (<100k)."""
        config = LGBMConfig.for_large_dataset(50_000)

        assert config.num_leaves == 31
        assert config.min_child_samples == 20

    def test_for_large_dataset_medium(self):
        """Test config for medium dataset (100k-1M)."""
        config = LGBMConfig.for_large_dataset(500_000)

        assert config.num_leaves == 63
        assert config.min_child_samples == 50
        assert config.subsample == 0.8

    def test_for_large_dataset_large(self):
        """Test config for large dataset (1M-10M)."""
        config = LGBMConfig.for_large_dataset(4_000_000)

        assert config.num_leaves == 127
        assert config.min_child_samples == 100
        assert config.max_bin == 511
        assert config.subsample == 0.7

    def test_for_large_dataset_very_large(self):
        """Test config for very large dataset (10M+)."""
        config = LGBMConfig.for_large_dataset(15_000_000)

        assert config.num_leaves == 255
        assert config.min_child_samples == 200
        assert config.subsample == 0.6


# =============================================================================
# LGBMModel Initialization Tests
# =============================================================================


class TestLGBMModelInit:
    """Tests for LGBMModel initialization."""

    def test_default_initialization(self):
        """Test model with default config."""
        model = LGBMModel()

        assert model.config is not None
        assert model.model is None
        assert model.feature_names is None
        assert model.best_iteration is None

    def test_custom_config_initialization(self):
        """Test model with custom config."""
        config = LGBMConfig(num_leaves=63, learning_rate=0.1)
        model = LGBMModel(config=config)

        assert model.config.num_leaves == 63
        assert model.config.learning_rate == 0.1

    def test_from_xgboost_params_factory(self):
        """Test factory method from XGBoost params."""
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
        }

        model = LGBMModel.from_xgboost_params(xgb_params)

        assert model.config.n_estimators == 300
        assert model.config.max_depth == 5


# =============================================================================
# LGBMModel Training Tests
# =============================================================================


class TestLGBMModelFit:
    """Tests for LGBMModel fit method."""

    def test_fit_with_dataframe(self, sample_train_data):
        """Test fitting with pandas DataFrame."""
        X, y = sample_train_data
        model = LGBMModel()

        result = model.fit(X, y, verbose=False)

        assert result is model  # Returns self
        assert model.model is not None
        assert model.feature_names == list(X.columns)

    def test_fit_with_numpy_arrays(self, sample_train_data):
        """Test fitting with numpy arrays."""
        X, y = sample_train_data
        model = LGBMModel()

        model.fit(X.values, y.values, verbose=False)

        assert model.model is not None
        assert model.feature_names is None  # Not available from numpy

    def test_fit_with_feature_names(self, sample_train_data):
        """Test fitting with explicit feature names."""
        X, y = sample_train_data
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        model = LGBMModel()

        model.fit(X.values, y.values, feature_names=feature_names, verbose=False)

        assert model.feature_names == feature_names

    def test_fit_with_validation_data(self, sample_train_val_data):
        """Test fitting with validation data for early stopping."""
        data = sample_train_val_data
        config = LGBMConfig(early_stopping_rounds=10, n_estimators=1000)
        model = LGBMModel(config=config)

        model.fit(
            data["X_train"], data["y_train"],
            X_val=data["X_val"], y_val=data["y_val"],
            verbose=False,
        )

        assert model.model is not None
        assert model.best_iteration is not None
        # Early stopping should stop before max iterations
        assert model.best_iteration < 1000

    def test_fit_stores_best_iteration(self, sample_train_val_data):
        """Test that best iteration is stored."""
        data = sample_train_val_data
        model = LGBMModel()

        model.fit(
            data["X_train"], data["y_train"],
            X_val=data["X_val"], y_val=data["y_val"],
            verbose=False,
        )

        assert model.best_iteration is not None
        assert isinstance(model.best_iteration, int)


# =============================================================================
# LGBMModel Prediction Tests
# =============================================================================


class TestLGBMModelPredict:
    """Tests for LGBMModel predict method."""

    def test_predict_returns_array(self, fitted_model, sample_train_data):
        """Test that predict returns numpy array."""
        X, _ = sample_train_data

        predictions = fitted_model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_predict_with_numpy_array(self, fitted_model, sample_train_data):
        """Test prediction with numpy array input."""
        X, _ = sample_train_data

        predictions = fitted_model.predict(X.values)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_predict_unfitted_raises_error(self, sample_train_data):
        """Test that predicting with unfitted model raises error."""
        X, _ = sample_train_data
        model = LGBMModel()

        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)

    def test_predict_reasonable_values(self, fitted_model, sample_train_data):
        """Test that predictions are in reasonable range."""
        X, y = sample_train_data

        predictions = fitted_model.predict(X)

        # Predictions should be roughly in the same range as target
        assert predictions.min() > y.min() - 3 * y.std()
        assert predictions.max() < y.max() + 3 * y.std()


class TestLGBMModelPredictWithInterval:
    """Tests for prediction intervals."""

    def test_predict_with_interval_returns_tuple(self, fitted_model, sample_train_data):
        """Test that predict_with_interval returns 3 arrays."""
        X, _ = sample_train_data

        result = fitted_model.predict_with_interval(X)

        assert isinstance(result, tuple)
        assert len(result) == 3
        predictions, lower, upper = result
        assert len(predictions) == len(X)
        assert len(lower) == len(X)
        assert len(upper) == len(X)

    def test_predict_interval_ordering(self, fitted_model, sample_train_data):
        """Test that lower <= prediction <= upper."""
        X, _ = sample_train_data

        predictions, lower, upper = fitted_model.predict_with_interval(X)

        assert np.all(lower <= predictions)
        assert np.all(predictions <= upper)

    def test_predict_interval_width(self, fitted_model, sample_train_data):
        """Test that interval width is positive."""
        X, _ = sample_train_data

        predictions, lower, upper = fitted_model.predict_with_interval(X)

        interval_width = upper - lower
        assert np.all(interval_width >= 0)


# =============================================================================
# LGBMModel Feature Importance Tests
# =============================================================================


class TestLGBMModelFeatureImportance:
    """Tests for feature importance methods."""

    def test_get_feature_importance_returns_dataframe(self, fitted_model):
        """Test that feature importance returns DataFrame."""
        importance = fitted_model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_get_feature_importance_all_features(self, fitted_model, sample_train_data):
        """Test that all features are included."""
        X, _ = sample_train_data
        importance = fitted_model.get_feature_importance()

        assert len(importance) == len(X.columns)

    def test_get_feature_importance_sorted(self, fitted_model):
        """Test that importance is sorted descending."""
        importance = fitted_model.get_feature_importance()

        values = importance["importance"].values
        assert np.all(values[:-1] >= values[1:])

    def test_get_feature_importance_unfitted_raises_error(self):
        """Test error when model not fitted."""
        model = LGBMModel()

        with pytest.raises(ValueError, match="Model not fitted"):
            model.get_feature_importance()

    def test_feature_importance_uses_feature_names(self, fitted_model, sample_train_data):
        """Test that feature names are used."""
        X, _ = sample_train_data
        importance = fitted_model.get_feature_importance()

        assert set(importance["feature"]) == set(X.columns)


# =============================================================================
# LGBMModel Save/Load Tests
# =============================================================================


class TestLGBMModelPersistence:
    """Tests for model save and load."""

    def test_save_creates_file(self, fitted_model, tmp_model_path):
        """Test that save creates a file."""
        fitted_model.save(tmp_model_path)

        assert tmp_model_path.exists()

    def test_save_creates_parent_directories(self, fitted_model, tmp_path):
        """Test that save creates parent directories."""
        nested_path = tmp_path / "nested" / "dir" / "model.pkl"

        fitted_model.save(nested_path)

        assert nested_path.exists()

    def test_load_restores_model(self, fitted_model, tmp_model_path, sample_train_data):
        """Test that load restores a working model."""
        X, _ = sample_train_data

        # Get predictions before save
        predictions_before = fitted_model.predict(X)

        # Save and load
        fitted_model.save(tmp_model_path)
        loaded_model = LGBMModel.load(tmp_model_path)

        # Get predictions after load
        predictions_after = loaded_model.predict(X)

        np.testing.assert_array_almost_equal(predictions_before, predictions_after)

    def test_load_restores_config(self, tmp_model_path, sample_train_data):
        """Test that load restores config."""
        X, y = sample_train_data
        config = LGBMConfig(num_leaves=63, learning_rate=0.1)
        model = LGBMModel(config=config)
        model.fit(X, y, verbose=False)

        model.save(tmp_model_path)
        loaded_model = LGBMModel.load(tmp_model_path)

        assert loaded_model.config.num_leaves == 63
        assert loaded_model.config.learning_rate == 0.1

    def test_load_restores_feature_names(self, fitted_model, tmp_model_path, sample_train_data):
        """Test that load restores feature names."""
        X, _ = sample_train_data

        fitted_model.save(tmp_model_path)
        loaded_model = LGBMModel.load(tmp_model_path)

        assert loaded_model.feature_names == list(X.columns)

    def test_load_restores_best_iteration(self, tmp_model_path, sample_train_val_data):
        """Test that load restores best_iteration."""
        data = sample_train_val_data
        model = LGBMModel()
        model.fit(
            data["X_train"], data["y_train"],
            X_val=data["X_val"], y_val=data["y_val"],
            verbose=False,
        )

        original_best = model.best_iteration
        model.save(tmp_model_path)
        loaded_model = LGBMModel.load(tmp_model_path)

        assert loaded_model.best_iteration == original_best


# =============================================================================
# LGBMModel Cross-Validation Tests
# =============================================================================


class TestLGBMModelCrossValidation:
    """Tests for cross-validation method."""

    def test_cross_validate_returns_dict(self, sample_train_data):
        """Test that cross_validate returns dictionary."""
        X, y = sample_train_data
        model = LGBMModel()

        scores = model.cross_validate(X, y, n_splits=3)

        assert isinstance(scores, dict)

    def test_cross_validate_contains_metrics(self, sample_train_data):
        """Test that CV results contain expected metrics."""
        X, y = sample_train_data
        model = LGBMModel()

        scores = model.cross_validate(X, y, n_splits=3)

        assert "rmse" in scores
        assert "mae" in scores
        assert "mape" in scores
        assert "rmse_mean" in scores
        assert "rmse_std" in scores

    def test_cross_validate_correct_fold_count(self, sample_train_data):
        """Test that correct number of folds are run."""
        X, y = sample_train_data
        model = LGBMModel()
        n_splits = 4

        scores = model.cross_validate(X, y, n_splits=n_splits)

        assert len(scores["rmse"]) == n_splits
        assert len(scores["mae"]) == n_splits

    def test_cross_validate_with_numpy(self, sample_train_data):
        """Test CV with numpy arrays."""
        X, y = sample_train_data
        model = LGBMModel()

        scores = model.cross_validate(X.values, y.values, n_splits=3)

        assert "rmse_mean" in scores
        assert scores["rmse_mean"] > 0


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_gpu_config(self):
        """Test GPU config generator."""
        gpu_config = get_gpu_config()

        assert isinstance(gpu_config, dict)
        assert gpu_config["device"] == "gpu"
        assert "gpu_platform_id" in gpu_config
        assert "gpu_device_id" in gpu_config

    def test_migrate_xgboost_to_lightgbm_basic(self):
        """Test basic XGBoost to LightGBM migration."""
        xgb_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
        }

        lgb_params = migrate_xgboost_to_lightgbm(xgb_params)

        assert lgb_params["n_estimators"] == 500
        assert lgb_params["max_depth"] == 6
        assert lgb_params["learning_rate"] == 0.1

    def test_migrate_xgboost_to_lightgbm_sets_defaults(self):
        """Test that migration sets LightGBM defaults."""
        xgb_params = {}

        lgb_params = migrate_xgboost_to_lightgbm(xgb_params)

        assert lgb_params["objective"] == "regression"
        assert lgb_params["metric"] == "rmse"
        assert lgb_params["boosting_type"] == "gbdt"

    def test_migrate_xgboost_to_lightgbm_subsample(self):
        """Test subsample -> bagging_fraction mapping."""
        xgb_params = {"subsample": 0.8}

        lgb_params = migrate_xgboost_to_lightgbm(xgb_params)

        assert lgb_params["bagging_fraction"] == 0.8
        assert lgb_params["bagging_freq"] == 1

    def test_migrate_xgboost_to_lightgbm_colsample(self):
        """Test colsample_bytree -> feature_fraction mapping."""
        xgb_params = {"colsample_bytree": 0.7}

        lgb_params = migrate_xgboost_to_lightgbm(xgb_params)

        assert lgb_params["feature_fraction"] == 0.7

    def test_migrate_xgboost_to_lightgbm_regularization(self):
        """Test regularization parameter mapping."""
        xgb_params = {
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
        }

        lgb_params = migrate_xgboost_to_lightgbm(xgb_params)

        assert lgb_params["reg_alpha"] == 0.5
        assert lgb_params["reg_lambda"] == 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestLGBMModelIntegration:
    """Integration tests for full workflow."""

    def test_full_training_workflow(self, sample_train_val_data):
        """Test complete training workflow."""
        data = sample_train_val_data

        # Create model with custom config
        config = LGBMConfig(
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=100,
            early_stopping_rounds=10,
        )
        model = LGBMModel(config=config)

        # Fit with validation
        model.fit(
            data["X_train"], data["y_train"],
            X_val=data["X_val"], y_val=data["y_val"],
            verbose=False,
        )

        # Predict
        predictions = model.predict(data["X_val"])

        # Get feature importance
        importance = model.get_feature_importance()

        # Verify
        assert len(predictions) == len(data["X_val"])
        assert len(importance) == len(data["X_train"].columns)

    def test_save_load_predict_workflow(self, sample_train_data, tmp_model_path):
        """Test save, load, and predict workflow."""
        X, y = sample_train_data

        # Train and save
        model = LGBMModel()
        model.fit(X, y, verbose=False)
        original_predictions = model.predict(X)
        model.save(tmp_model_path)

        # Load and predict
        loaded_model = LGBMModel.load(tmp_model_path)
        loaded_predictions = loaded_model.predict(X)

        # Verify predictions match
        np.testing.assert_array_almost_equal(
            original_predictions,
            loaded_predictions,
        )

    def test_model_improves_over_baseline(self, sample_train_val_data):
        """Test that model predictions are better than mean baseline."""
        data = sample_train_val_data

        model = LGBMModel()
        model.fit(
            data["X_train"], data["y_train"],
            X_val=data["X_val"], y_val=data["y_val"],
            verbose=False,
        )

        predictions = model.predict(data["X_val"])
        y_val = data["y_val"].values

        # Model RMSE
        model_rmse = np.sqrt(np.mean((y_val - predictions) ** 2))

        # Baseline RMSE (predict mean)
        baseline_rmse = np.sqrt(np.mean((y_val - np.mean(data["y_train"])) ** 2))

        # Model should be better than baseline
        assert model_rmse < baseline_rmse
