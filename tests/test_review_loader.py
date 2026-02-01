from modules.word_freq.ReviewLoader import ReviewLoader


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
