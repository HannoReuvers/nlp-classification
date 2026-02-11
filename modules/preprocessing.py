import pandas as pd
import numpy as np
from pathlib import Path
import re
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer

random_state=42
np.random.seed(random_state)

# custom nltk directory
NLTK_DATA = Path.home() / "nltk_data"
nltk.data.path.append(str(NLTK_DATA))

RESOURCES = [
    "punkt",
    "stopwords",
    "averaged_perceptron_tagger",
    "wordnet",
]

for res in RESOURCES:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res, download_dir=str(NLTK_DATA), quiet=True)

tokenizer = nltk.word_tokenize
stop_words = set(nltk.corpus.stopwords.words("english"))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


class TextPreprocessor:

    def __init__(
        self,
        df: pd.DataFrame,
        column: str,
        stop_words: Optional[set[str]] = None,
        tokenizer: Optional[Callable] = None,
        stemmer: Optional[Callable] = None,
        lemmatizer: Optional[Callable] = None,
    ):
        self.df = df
        self.column = column
        self.stop_words = stop_words or set()
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.lemmatizer = lemmatizer

    def clean(self, tokenize: bool = True):

        clean_series = (
            self.df[self.column]
            .str.casefold()
            .str.replace("<br />", "\n", regex=False)
            .str.replace(r"[^a-zA-Z\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        if tokenize:
            if self.tokenizer:
                clean_series = clean_series.map(self.tokenizer)
            else:
                clean_series = clean_series.str.split()

        self.df[self.column] = clean_series

        return self.df

    def remove_stopwords(self):

        if not self.stop_words:
            return self.df

        self.df[self.column] = self.df[self.column].map(
            lambda tokens: [t for t in tokens if t not in self.stop_words]
        )

        return self.df
    
    def _get_wordnet_pos(self, tag: str) -> str:
        tag_dict = {"J": "a", "N": "n", "V": "v", "R": "r"}

        return tag_dict.get(tag[0].upper(), "n")

    def _stem_tokens(self, tokens: list[str]) -> list[str]:

        return [self.stemmer.stem(t) for t in tokens]

    def _lemm_tokens(self, tokens: list[str]) -> list[str]:
        pos_tags = nltk.pos_tag(tokens)

        return [
            self.lemmatizer.lemmatize(word, self._get_wordnet_pos(tag))
            for word, tag in pos_tags
        ]

    def stem_or_lemmatize(self, method: Literal["stem", "lemm"] = "lemm"):

        col = self.df[self.column]

        if method == "stem" and self.stemmer:
            self.df[self.column] = col.map(self._stem_tokens)

        elif method == "lemm" and self.lemmatizer:
            self.df[self.column] = col.map(self._lemm_tokens)

        else:
            raise ValueError("Invalid method or missing stemmer/lemmatizer")

        return self.df

    def process(
        self,
        tokenize: bool = True,
        remove_stop: bool = True,
        method: Optional[Literal["stem", "lemm"]] = "lemm",
    ):

        self.clean(tokenize)

        if remove_stop:
            self.remove_stopwords()

        if method:
            self.stem_or_lemmatize(method)

        return self.df