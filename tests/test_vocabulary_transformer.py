from modules.word_freq.VocabularyTransformer import VocabularyTransformer


class TestVocabularyTransformer:
    """Test suite for class VocabularyTransformer"""

    def test_vocabulary_transformer_initialization(self):
        """Test that ReviewLoader initializes correctly."""
        transformer = VocabularyTransformer()
        assert transformer.vocabulary_size == 5000
        assert transformer.method == "frequency"
