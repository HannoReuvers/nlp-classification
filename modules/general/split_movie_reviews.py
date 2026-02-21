import csv
from pathlib import Path
from typing import Literal
import random

from src.config import Config


class SplitMovieReviews:
    def __init__(self, config: Config = Config()):
        self.config = config

    def split_reviews(self, validation_size: int = 2500, random_seed: int = 42) -> None:
        def reviews_to_csv(
            review_list: list[str],
            output_file: str,
            label: int,
            write_mode: Literal["w", "a"],
        ) -> None:
            with open(output_file, mode=write_mode, encoding="utf-8") as output_file:
                writer = csv.writer(output_file)

                # Include header if write mode is "w" (write), but not if "a" (append)
                if write_mode == "w":
                    writer.writerow(["label", "review_id", "review_text"])

                # Write reviews to CSV ("w" for write, "a" for append)
                for review_path in review_list:
                    with open(review_path, mode="r", encoding="utf-8") as review_file:
                        review_text = review_file.read()
                        writer.writerow([label, Path(review_path).name, review_text])

        # Configuration
        review_types = ["neg", "pos"]
        labels = [0, 1]
        write_modes = ["w", "a"]

        # ------------------ TRAIN DATA ------------------#

        # Output file name
        train_output_file = self.config.DATA_SPLIT_DIR / "train_reviews.csv"

        # Write to CSV
        for label, review_type, write_mode in zip(labels, review_types, write_modes):
            train_reviews_path = self.config.INPUT_DATA_DIR / f"train/{review_type}"
            train_reviews = list(train_reviews_path.glob("*.txt"))

            reviews_to_csv(
                train_reviews, train_output_file, label=label, write_mode=write_mode
            )

        # ------------------ VALIDATION AND TEST DATA ------------------#

        # Set random seed for reproducibility
        random.seed(random_seed)

        # Output file names
        validation_output_file = self.config.DATA_SPLIT_DIR / "validation_reviews.csv"
        test_output_file = self.config.DATA_SPLIT_DIR / "test_reviews.csv"

        # Write to CSV
        for label, review_type, write_mode in zip(labels, review_types, write_modes):
            # Source (old test data)
            source_folder = self.config.INPUT_DATA_DIR / f"test/{review_type}"

            # Fetch all reviews from old test data
            all_reviews = list(source_folder.glob("*.txt"))

            # Write validation reviews to CSV (random sample with validation_size records)
            validation_reviews = random.sample(all_reviews, validation_size)
            reviews_to_csv(
                validation_reviews,
                validation_output_file,
                label=label,
                write_mode=write_mode,
            )

            # Write test reviews to CSV ()
            test_reviews = [
                review for review in all_reviews if review not in validation_reviews
            ]
            reviews_to_csv(
                test_reviews, test_output_file, label=label, write_mode=write_mode
            )

        """
        input_path = self.config.raw_data_path
        output_path = self.config.processed_data_path

        # Create output directories (if needed)
        Path(output_path + "/validation/pos").mkdir(parents=True, exist_ok=True)
        Path(output_path + "/validation/neg").mkdir(parents=True, exist_ok=True)
        Path(output_path + "/test/pos").mkdir(parents=True, exist_ok=True)
        Path(output_path + "/test/neg").mkdir(parents=True, exist_ok=True)

        # Train data remains as-is
        shutil.copytree(input_path + "/aclImdb/train/pos", output_path + "/train/pos")
        shutil.copytree(input_path + "/aclImdb/train/neg", output_path + "/train/neg")

        # Split old test data into validation and test sets
        review_type_list = ["pos", "neg"]
        for review_type in review_type_list:
            # Source (old test data)
            source_folder = input_path + "/aclImdb/test/" + review_type

            # Destination
            dest_validation_folder = output_path + "/validation/" + review_type
            dest_test_folder = output_path + "/test/" + review_type

            # Fetch all reviews from old test data
            all_reviews = os.listdir(source_folder)
            print(f"Number of {review_type} reviews: {len(all_reviews)}")

            # Reviews for validation set
            random.seed(42)
            validation_reviews = random.sample(all_reviews, validation_size)

            for review in all_reviews:
                if review in validation_reviews:
                    shutil.copyfile(
                        source_folder + "/" + review, dest_validation_folder + "/" + review
                    )
                else:
                    shutil.copyfile(
                        source_folder + "/" + review, dest_test_folder + "/" + review
                    )
                    """
