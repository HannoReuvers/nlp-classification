from collections import Counter

import pandas as pd
from tqdm import tqdm
from modules.bag_of_words.ReviewLoader import ReviewLoader


class VocabularyTransformer:
    def __init__(self, vocabulary_size: int = 5000):
        self.vocabulary_size = vocabulary_size

    def most_common_words(self, review_df: pd.DataFrame, stopwords: list) -> dict:
        """
        Fit the vocabulary transformer to the reviews in the review_df dataframe.

        Args:
            review_df (pd.Dataframe): Dataframe containing the reviews. The column containing the review text should be named "review_text".
            stopwords (list): List of stop words to exclude.

        Returns:
            dict: A dictionary mapping words to integer indices.
        """
        overall_counter = Counter()

        # Scan through reviews and build word frequency counter
        for _, row in tqdm(
            review_df.iterrows(),
            total=len(review_df),
            desc="Constructing vocabulary",
            leave=False,
        ):
            # Fetch review text
            review_text = row["review_text"]

            # Process review text with ReviewLoader logic
            loader = ReviewLoader()
            loader.load_review(review_text)
            loader.create_word_counter(stopwords)

            # Accumulate word counts
            overall_counter += loader.word_counter

        # Select most common words based on vocabulary size
        most_common_word_counts = overall_counter.most_common(self.vocabulary_size)
        count_dict = dict(most_common_word_counts)

        return count_dict

    def fit(self, review_df: pd.DataFrame, stopwords: list) -> dict:
        """
        Fit the vocabulary transformer to the reviews in the review_df dataframe.

        Args:
            review_df (pd.Dataframe): Dataframe containing the reviews. The column containing the review text should be named "review_text".
            stopwords (list): List of stop words to exclude.

        Returns:
            vocabulary: A dictionary mapping words to integer indices.
        """
        most_common_word_counts = self.most_common_words(review_df, stopwords)

        # Create vocabulary mapping
        vocabulary = {
            word: idx + 1
            for idx, (word, _) in enumerate(most_common_word_counts.items())
        }

        return vocabulary

    def transform(
        self, review_df: pd.DataFrame, vocabulary: dict, print_name: str
    ) -> pd.DataFrame:
        """
        Transform all reviews in the review_df dataframe into an integer representation. This integer
        representation is based on the provided vocabulary.

        Args:
            review_df (pd.Dataframe): Dataframe containing the reviews. The column containing the review text should be named "review_text".
            vocabulary (dict): Dictionary mapping words to integer indices.
            label (int): Label to assign to all transformed reviews.
        """

        label_list = []
        review_id_list = []
        word_sequence_list = []

        for _, row in tqdm(
            review_df.iterrows(),
            total=len(review_df),
            desc=f"Processing {print_name} reviews",
            leave=False,
        ):
            # Fetch review text
            review_text = row["review_text"]

            # Transform review to integer list
            loader = ReviewLoader()
            loader.load_review(review_text)
            int_list = loader.review_to_integer_mapping([], vocabulary)
            int_string = "-".join(map(str, int_list))

            # Append results to lists
            label_list.append(row["label"])
            review_id_list.append(row["review_id"])
            word_sequence_list.append(int_string)

        # Create output pandas DataFrame
        output_df = pd.DataFrame(
            {
                "label": label_list,
                "review_id": review_id_list,
                "word_sequence": word_sequence_list,
            }
        )

        return output_df
