"Configuration module"

import logging
from pathlib import Path
from typing import Final
import warnings


class Config:
    # Paths
    BASE_DIR: Final = Path(__file__).parent.parent
    # MLFLOW_DIR: Final = BASE_DIR / "mlflow"
    MLFLOW_DIR: Final = BASE_DIR
    MLFLOW_TRACKING_URI: Final = MLFLOW_DIR / "mlflow-runs"

    # Destination folder for train, validation, and test data
    INPUT_DATA_DIR: Final = BASE_DIR / "data/0_raw/aclImdb"
    DATA_SPLIT_DIR: Final = BASE_DIR / "data/1_data_split"
    DATA_MODEL_INPUT_DIR: Final = BASE_DIR / "data/2_model_input"

    # Bag of Words configuration
    VOCABULARY_SIZE: Final = 5000
    VOCAB_DIR: Final = BASE_DIR / "data/vocabularies"

    # Logger configuration
    # Suppress MLflow verbose output
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("mlflow.tracking").setLevel(logging.WARNING)
    logging.getLogger("mlflow.store").setLevel(logging.WARNING)
    logging.getLogger("mlflow.models").setLevel(logging.WARNING)
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    # Suppress scikit-learn FutureWarning about pickle format
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*pickle.*")

    @classmethod
    def check_directory_presence(cls) -> None:
        assert (
            cls.DATA_SPLIT_DIR.exists()
        ), f"Directory {cls.DATA_SPLIT_DIR} does not exist"
        assert (
            cls.DATA_MODEL_INPUT_DIR.exists()
        ), f"Directory {cls.DATA_MODEL_INPUT_DIR} does not exist"
