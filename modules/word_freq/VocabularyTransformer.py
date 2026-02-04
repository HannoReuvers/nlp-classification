from collections import Counter
import os

import pandas as pd
from tqdm import tqdm
from modules.word_freq.ReviewLoader import ReviewLoader


class VocabularyTransformer:
    def __init__(self, vocabulary_size: int = 5000):
        self.vocabulary_size = vocabulary_size

    def most_common_words(self, path_list: list, stopwords: list) -> dict:
        """
        Fit the vocabulary transformer to the provided list of review file paths.

        Args:
            path_list (list): List of file paths containing reviews.
            stopwords (list): List of stop words to exclude.

        Returns:
            dict: A dictionary mapping words to integer indices.
        """
        overall_counter = Counter()

        # Scan through reviews and build word frequency counter
        for review_path in tqdm(path_list):
            loader = ReviewLoader()
            loader.load_review(review_path)
            loader.create_word_counter(stopwords)

            # Accumulate word counts
            overall_counter += loader.word_counter

        # Select most common words based on vocabulary size
        most_common_word_counts = overall_counter.most_common(self.vocabulary_size)
        count_dict = dict(most_common_word_counts)

        return count_dict

    def fit(self, path_list: list, stopwords: list) -> dict:
        """
        Fit the vocabulary transformer to the provided list of review file paths.

        Args:
            path_list (list): List of file paths containing reviews.
            stopwords (list): List of stop words to exclude.

        Returns:
            vocabulary: A dictionary mapping words to integer indices.
        """
        most_common_word_counts = self.most_common_words(path_list, stopwords)

        # Create vocabulary mapping
        vocabulary = {
            word: idx + 1
            for idx, (word, _) in enumerate(most_common_word_counts.items())
        }

        return vocabulary

    def transform(self, path_list: list, vocabulary: dict, label: int) -> pd.DataFrame:
        """
        Transform all reviews in the list of path into an integer representation. This integer
        representation is based on the provided vocabulary.

        Args:
            path_list (list): List of file paths containing reviews.
            vocabulary (dict): Dictionary mapping words to integer indices.
            label (int): Label to assign to all transformed reviews.
        """

        file_name_list = []
        word_sequence_list = []

        for review_path in tqdm(path_list):
            # File name
            filename = os.path.basename(review_path)
            file_name_list.append(filename)

            # Transform review to integer list
            loader = ReviewLoader()
            loader.load_review(review_path)
            int_list = loader.review_to_integer_mapping([], vocabulary)
            int_string = "-".join(map(str, int_list))
            word_sequence_list.append(int_string)

        label_list = [label] * len(file_name_list)

        # Create output pandas DataFrame
        output_df = pd.DataFrame(
            {
                "file_name": file_name_list,
                "label": label_list,
                "word_sequence": word_sequence_list,
            }
        )

        return output_df
