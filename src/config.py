"Configuration module"

from pathlib import Path
from typing import Final


class Config:
    # Paths
    BASE_DIR: Final = Path(__file__).parent.parent

    # Destination folder for train, validation, and test data
    INPUT_DATA_DIR: Final = BASE_DIR / "data/0_raw/aclImdb"
    DATA_SPLIT_DIR: Final = BASE_DIR / "data/1_data_split"
    DATA_MODEL_INPUT_DIR: Final = BASE_DIR / "data/2_model_input"

    @classmethod
    def check_directory_presence(cls) -> None:
        assert (
            cls.DATA_SPLIT_DIR.exists()
        ), f"Directory {cls.DATA_SPLIT_DIR} does not exist"
        assert (
            cls.DATA_MODEL_INPUT_DIR.exists()
        ), f"Directory {cls.DATA_MODEL_INPUT_DIR} does not exist"

    """
    DATA_DIR: Final = "data"
    RAW_DATA_DIR: Final = f"{DATA_DIR}/0_raw_data"
    SPLIT_DATA_DIR: Final = f"{DATA_DIR}/1_data_split"
    VOCAB_DIR: Final = f"{DATA_DIR}/vocabularies"
    MODEL_INPUT_DIR: Final = f"{DATA_DIR}/2_model_input"

    # Vocabulary settings
    VOCABULARY_SIZE: Final = 5000

    # NLP settings
    STOPWORDS_FILE: Final = "stopwords.txt"

    @staticmethod
    def create_directories() -> None:
        #Create necessary directories for data storage.
        Path(Config.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.SPLIT_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.VOCAB_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.MODEL_INPUT_DIR).mkdir(parents=True, exist_ok=True)
    """
