from modules.word_freq.VocabularyTransformer import VocabularyTransformer


class TestVocabularyTransformer:
    """Test suite for class VocabularyTransformer"""

    def test_vocabulary_transformer_initialization(self):
        """Test that ReviewLoader initializes correctly."""
        transformer = VocabularyTransformer()
        assert transformer.vocabulary_size == 5000

    def test_fit_method(self, multiple_review_paths):
        """
        Test that fit method creates a vocabulary mapping from multiple review files.

        Args:
            multiple_review_paths: Fixture providing multiple mock review files.
        """
        transformer = VocabularyTransformer(vocabulary_size=3)
        vocabulary = transformer.most_common_words(multiple_review_paths, stopwords=[])

        assert isinstance(vocabulary, dict)
        assert (
            len(vocabulary) <= 3
        )  # Vocabulary should not exceed requested vocabulary size (top3)
        assert vocabulary.get("movie") == 4  # "movie" appears in all 4 reviews
