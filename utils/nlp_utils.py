import csv
import os


def files_in_directory(directory: str) -> list:
    """Get a list of all file paths in a directory (non-recursive).

    Scans the specified directory and returns the full paths of all files found.
    Subdirectories are excluded from the results.

    Parameters
    ----------
    directory : str
        The path to the directory to scan. Can be an absolute or relative path.

    Returns
    -------
    list
        A list of strings, where each string is the absolute path to a file in
        the directory. Returns an empty list if the directory contains no files
        or only subdirectories.
    """
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file))
    ]


def read_vocabulary_as_dict(vocabulary_path: str, vocabulary_filename: str) -> dict:
    """Read vocabulary from a CSV file and return it as a dictionary.

    Expects a CSV file with 'word' and 'idx' columns where each row contains
    a word and its corresponding index.

    Parameters
    ----------
    vocabulary_path : str
        Directory path containing the vocabulary file.
    vocabulary_filename : str
        Name of the vocabulary CSV file.

    Returns
    -------
    dict
        Dictionary mapping words (str) to their indices (int).
    """
    vocabulary = {}
    vocabulary_path = os.path.join(vocabulary_path, vocabulary_filename)
    with open(vocabulary_path, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            vocabulary[row["word"]] = int(row["idx"])
    return vocabulary
