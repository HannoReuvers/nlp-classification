import pytest
from pathlib import Path


@pytest.fixture
def single_sentence_review(tmp_path):
    """
    Create a temporary file with a positive movie review.
    
    Args:
        tmp_path: pytest fixture providing a temporary directory path.
        
    Returns:
        Path: Path to the temporary review file.
    """
    review_content = "I have watched the movie."
    review_file = tmp_path / "positive_review.txt"
    review_file.write_text(review_content)
    return review_file
