from collections import Counter
import nltk
import re


class ReviewLoader:
    def __init__(self):
        self.review_content = None

    def load_review(self, review_path: str) -> None:
        """
        Load a movie review from the file specified by review_path.

        Args:
            review_path (str): Path to the movie review file.
        """
        try:
            with open(review_path, "r") as file:
                # Read review
                text_review = file.read()
            # Assign instance variables
            self.review_content = text_review
        except IOError:
            print(f"Unable to read file: {review_path}")

    def review_to_word_list(self, stopwords: list) -> None:
        """
        Creates a list of words from the original review text, removing punctuation and stop words.

        Args:
            stopwords (list): List of stop words.
        """
        review_without_punctuation = re.sub(r'[".,!?;-]+', "", self.original_review)
        review_tokenized = nltk.word_tokenize(review_without_punctuation)
        review_lower_case = [ch.lower() for ch in review_tokenized if ch.isalpha()]
        self.review_wordlist = [
            word for word in review_lower_case if word not in stopwords
        ]

    def create_word_counter(self, stopwords: list) -> None:
        """
        Creates a word frequency counter from the review word list.

        Args:
            stopwords (list): List of stop words.
        """
        if self.review_wordlist is None:
            self.review_to_wordlist(stopwords)

        self.word_counter = Counter()
        for word in self.review_wordlist:
            self.word_counter[word] = self.word_counter.get(word, 0) + 1

    def review_to_integer_mapping(self, stopwords: list, vocabulary) -> list:
        """
        Returns a list of integers representing the words in the review based on a given vocabulary mapping.

        Args:
            stopwords (list): List of stop words.
            vocabulary (dict):
        """

        if self.review_wordlist is None:
            self.review_to_wordlist(stopwords)

        int_list = []
        for word in self.review_wordlist:
            # Find integer representation of word and append to sequence
            word_int = vocabulary.get(word, 0)
            int_list.append(word_int)

        return int_list

    def __str__(self):
        return f"REVIEW:\n{self.review_content}"

    def get_word_counter(self):
        return self.word_counter
