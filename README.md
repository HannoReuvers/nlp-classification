# NLP-Classification


## Installation
1. Install the [uv Python package and project manager](https://docs.astral.sh/uv/)
2. At the moment of writing, the raw data is freely available on https://ai.stanford.edu/~amaas/data/sentiment/. Extract the data into the `data/raw/` subdirectory
3. Split the original test data into a validation and test data set by running

```
uv run scripts/0_split_test_data.py
```

All subsequent steps depend on your model choice. The table below offers a performance overview. More extensive descriptions can be found in the wiki.
