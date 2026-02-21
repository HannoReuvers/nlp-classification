from modules.general.split_movie_reviews import SplitMovieReviews
from src.config import Config


def main():
    try:
        Config.check_directory_presence()
    except Exception as e:
        raise Exception(f"An error occurred while checking directories: {e}")

    splitter = SplitMovieReviews()
    try:
        splitter.split_reviews()
        print("Movie reviews split successfully.")
    except Exception as e:
        raise Exception(f"An error occurred while splitting reviews: {e}")

    print("Hello from nlp-classification!")


if __name__ == "__main__":
    main()
