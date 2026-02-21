import pandas as pd
import pytest


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
    return review_content


@pytest.fixture
def multiple_review_paths(tmp_path):
    """
    Create multiple temporary files with movie reviews.

    Args:
        tmp_path: pytest fixture providing a temporary directory path.

    Returns:
        List[Path]: List of paths to the temporary review files.
    """
    reviews = [
        "The movie was fantastic!",
        "I absolutely loved the movie.",
        "The movie was terrible.",
        "I did not enjoy the movie at all.",
    ]
    return pd.DataFrame({"review_text": reviews})
