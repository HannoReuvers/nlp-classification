from collections import Counter

from modules.word_freq.ReviewLoader import ReviewLoader


class VocabularyTransformer:
    def __init__(self, vocabulary_size: int = 5000, method: str = "frequency"):
        self.vocabulary_size = vocabulary_size
        self.method = method

    def fit(self, path_list: list, stopwords: list) -> dict:
        """
        Fit the vocabulary transformer to the provided list of review file paths.

        Args:
            path_list (list): List of file paths to reviews.
            stopwords (list): List of stop words to exclude.

        Returns:
            dict: A dictionary mapping words to integer indices.
        """
        overall_counter = Counter()

        # Scan through reviews and build word frequency counter
        for review_path in path_list:
            loader = ReviewLoader()
            loader.load_review(review_path)
            loader.create_word_counter(list_stopwords=[])

            # Accumulate word counts
            overall_counter += loader.word_counter

        # Select most common words based on vocabulary size
        most_common = overall_counter.most_common(self.vocabulary_size)

        # Create vocabulary mapping
        vocabulary = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}

        return vocabulary
