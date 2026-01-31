# NLP-Classification


## Installation
1. Install the [uv Python package and project manager](https://docs.astral.sh/uv/)
2. At the moment of writing, the raw data is freely available on https://ai.stanford.edu/~amaas/data/sentiment/. Extract the data into the `data/raw/` subdirectory
3. Subsequently split the original test data into a validation and test data set by running

```
uv run modules/split_test_data.py
```
