"""
Module for splitting test data.

This module provides functionality to split test data for NLP classification tasks.
"""

import os
from pathlib import Path
import random
import shutil


def split_data(
    input_path: str,
    output_path: str,
    validation_size: int = 2500,
) -> None:
    """
    Split test data into train and validation sets.

    Args:
        input_path: Path to the input data directory
        output_path: Path to the output data directory
        validation_size: Sample size of the resulting validation dataset
    """

    # Create output directories (if needed)
    #Path(output_path+"/train/pos").mkdir(parents=True, exist_ok=True)
    #Path(output_path+"/train/neg").mkdir(parents=True, exist_ok=True)
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

        # Deter
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


def main() -> None:
    """
    Main function to execute the data splitting script.
    """

    # Define paths
    input_dir = "./data/0_raw"
    output_dir = "./data/1_input"

    # Execute splitting
    try:
        split_data(input_dir, output_dir)
        print("Data splitting completed successfully.")
    except Exception as e:
        raise Exception(f"An error occurred during data splitting: {e}")


if __name__ == "__main__":
    main()
