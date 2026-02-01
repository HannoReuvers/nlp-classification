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
    review_file = tmp_path / "review_0.txt"
    review_file.write_text(review_content)
    return review_file


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
        ("review_1.txt", "The movie was fantastic!"),
        ("review_2.txt", "I absolutely loved the movie."),
        ("review_3.txt", "The movie was terrible."),
        ("review_4.txt", "I did not enjoy the movie at all."),
    ]

    file_paths = []
    for file_name, content in reviews:
        review_file = tmp_path / file_name
        review_file.write_text(content)
        file_paths.append(review_file)

    return file_paths
