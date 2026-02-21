import logging
from modules.bag_of_words.BagOfWordsEstimator import BagOfWordsEstimator
import pandas as pd
from src.config import Config


def BagOfWordsModel():
    # Configure logging
    logger = logging.getLogger("BagOfWords ")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("#" * 50)
    logger.info("BAG OF WORDS")
    logger.info("#" * 50)

    """
    try:
        Config.check_directory_presence()
    except Exception as e:
        raise Exception(f"An error occurred while checking directories: {e}")

    # STEP 1
    logger.info("STEP 1: Split dataset into train, validation, and test data")
    try:
        splitter = SplitMovieReviews()
        splitter.split_reviews()
        logger.info("STEP 1: DONE")
    except Exception as e:
        logger.error(f"Error while splitting reviews: {e}")
        raise

    # STEP 2
    logger.info("STEP 2: Create vocabulary from training data")
    try:
        vocab_transformer = VocabularyTransformer(vocabulary_size=Config.VOCABULARY_SIZE)
        train_reviews = pd.read_csv(Config.DATA_SPLIT_DIR / "train_reviews.csv")
        stopwords = nltk.corpus.stopwords.words("english")
        vocabulary = vocab_transformer.fit(train_reviews,
                                           stopwords=stopwords)
        logger.info("STEP 2: DONE")
    except Exception as e:
        logger.error(f"Error while creating vocabulary: {e}")
        raise

    # STEP 3
    logger.info("STEP 3: Tokenize reviews")
    try:
        for dataset in ["train", "validation", "test"]:
            reviews_df = pd.read_csv(Config.DATA_SPLIT_DIR / f"{dataset}_reviews.csv")
            tokenized_df = vocab_transformer.transform(
                reviews_df, vocabulary, print_name=dataset
            )
            tokenized_df.to_csv(
                Config.DATA_MODEL_INPUT_DIR / f"{dataset}_reviews_tokenized.csv",
                index=False,
            )
        logger.info("STEP 3: DONE")
    except Exception as e:
        logger.error(f"Error while tokenizing reviews: {e}")
        raise
    """

    # STEP 4
    logger.info("STEP 4: Train and evaluate model")
    try:
        train_df = pd.read_csv(
            Config.DATA_MODEL_INPUT_DIR / "train_reviews_tokenized.csv"
        )
        validation_df = pd.read_csv(
            Config.DATA_MODEL_INPUT_DIR / "validation_reviews_tokenized.csv"
        )
        test_df = pd.read_csv(
            Config.DATA_MODEL_INPUT_DIR / "test_reviews_tokenized.csv"
        )
        BoW_model = BagOfWordsEstimator(
            df_train=train_df,
            df_validation=validation_df,
            df_test=test_df,
            mlflow_experiment="BagOfWords",
            run_name="BoW_Run",
            n_trials=10,
            config=Config(),
        )
        BoW_model.estimate_BoW_model()
        logger.info("STEP 4: DONE")
    except Exception as e:
        logger.error(f"Error while training and evaluating model: {e}")
        raise


if __name__ == "__main__":
    BagOfWordsModel()
