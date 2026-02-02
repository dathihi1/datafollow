"""Machine learning models for traffic prediction."""

from src.models.sarima import SARIMAModel, SARIMAConfig
from src.models.lgbm_model import (
    LGBMModel,
    LGBMConfig,
    get_gpu_config,
    migrate_xgboost_to_lightgbm,
)
from src.models.xgboost_model import XGBoostModel

# Prophet is optional (requires pip install prophet)
try:
    from src.models.prophet_model import ProphetModel, ProphetConfig
    PROPHET_AVAILABLE = True
except ImportError:
    ProphetModel = None
    ProphetConfig = None
    PROPHET_AVAILABLE = False

__all__ = [
    "SARIMAModel",
    "SARIMAConfig",
    "ProphetModel",
    "ProphetConfig",
    "LGBMModel",
    "LGBMConfig",
    "XGBoostModel",
    "PROPHET_AVAILABLE",
    # Migration helpers
    "get_gpu_config",
    "migrate_xgboost_to_lightgbm",
]
