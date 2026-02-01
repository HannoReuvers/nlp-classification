import csv
import os

import nltk

from modules.word_freq.VocabularyTransformer import VocabularyTransformer


def main(vocabulary_filename: str = "vocabulary.csv") -> None:
    # Default output directory
    OUTPUT_DIR = "data/vocabularies"

    # Collect all training review file paths
    def files_in_directory(directory: str) -> list:
        return [
            os.path.join(directory, file)
            for file in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, file))
        ]

    training_reviews_path = files_in_directory(
        "data/1_input/train/pos"
    ) + files_in_directory("data/1_input/train/neg")

    # Create vocabulary
    stopwords = nltk.corpus.stopwords.words("english")
    transformer = VocabularyTransformer(vocabulary_size=1000)
    vocabulary = transformer.fit(training_reviews_path, stopwords=stopwords)

    # Output file
    output_file = os.path.join(OUTPUT_DIR, vocabulary_filename)

    # Store data as CSV
    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["word", "idx"])  # header
        for word, idx in vocabulary.items():  # vocabulary entries
            writer.writerow([word, idx])

    print(f"Vocabulary saved to {output_file}")


if __name__ == "__main__":
    main()
