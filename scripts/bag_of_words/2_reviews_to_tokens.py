from pathlib import Path

import pandas as pd

from modules.word_freq.VocabularyTransformer import VocabularyTransformer
from utils.nlp_utils import files_in_directory, read_vocabulary_as_dict


def main(vocabulary_filename: str = "vocabulary.csv") -> None:
    # Default directories
    VOCAB_DIR = "data/vocabularies"
    OUTPUT_DIR = "data/2_model_input"

    # Create output directories (if needed)
    Path(OUTPUT_DIR + "/train").mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR + "/validation").mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR + "/test").mkdir(parents=True, exist_ok=True)

    # Read the vocabulary
    vocabulary = read_vocabulary_as_dict(VOCAB_DIR, vocabulary_filename)

    # Generate tokenized reviews for training, validation, and test sets
    for dataset in ["train", "validation", "test"]:
        # Inform user
        print(f"\nProcessing {dataset} dataset...")

        # Output folder
        output_folder = f"{OUTPUT_DIR}/{dataset}"
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Init transformer
        voc_transformer = VocabularyTransformer()
        result_df = pd.DataFrame()
        for review_type in ["pos", "neg"]:
            # Map review type to label
            if review_type == "pos":
                label = 1
            elif review_type == "neg":
                label = 0

            # Transform reviews to tokenized format and store in df
            reviews_path = files_in_directory(
                f"data/1_data_split/{dataset}/{review_type}"
            )
            temp_df = voc_transformer.transform(
                reviews_path, vocabulary=vocabulary, label=label
            )
            result_df = pd.concat([result_df, temp_df], ignore_index=True)

        # Store result
        output_file = f"{output_folder}/{dataset}_reviews_tokenized.csv"
        result_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
