from collections import Counter
from modules.bag_of_words.ReviewLoader import ReviewLoader


class TestReviewLoader:
    """Test suite for class ReviewLoader"""

    def test_review_loader_initialization(self):
        """Test that ReviewLoader initializes correctly."""
        loader = ReviewLoader()
        assert loader.review_content is None

    def test_single_sentence_review(self, single_sentence_review):
        """
        Test that load_review successfully loads a review file.

        Args:
            single_sentence_review_file: Fixture providing mock review.
        """
        loader = ReviewLoader()
        loader.load_review(str(single_sentence_review))

        assert loader.review_content is not None
        assert len(loader.review_content) > 0
        assert loader.review_content == "I have watched the movie."

    def test_review_to_word_list(self, single_sentence_review):
        """
        Test that review_to_word_list creates a list of words from the review text, removing punctuation and stop words.

        Args:
            single_sentence_review_file: Fixture providing mock review.
        """
        loader = ReviewLoader()
        loader.load_review(str(single_sentence_review))
        loader.create_word_counter(stopwords=["i", "have", "the"])

        assert loader.word_counter is not None
        assert isinstance(loader.word_counter, Counter)
        assert loader.word_counter.get("watched") == 1
        assert loader.word_counter.get("movie") == 1
        assert loader.get_word_counter().get("watched") == 1
        assert loader.get_word_counter().get("movie") == 1

    def test_review_to_integer_mapping(self, single_sentence_review):
        """
        Test that review_to_integer_mapping returns a list of integers representing the words in the review based on a given vocabulary mapping.

        Args:
            single_sentence_review_file: Fixture providing mock review.
        """
        loader = ReviewLoader()
        loader.load_review(str(single_sentence_review))
        loader.create_word_counter(stopwords=["i", "have", "the"])

        vocabulary = {"watched": 1, "movie": 2}
        integer_mapping = loader.review_to_integer_mapping(
            stopwords=["i", "have", "the"], vocabulary=vocabulary
        )

        assert isinstance(integer_mapping, list)
        assert integer_mapping == [1, 2]
