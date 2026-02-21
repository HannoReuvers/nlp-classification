# NLP-Classification


## Installation
1. Install the [uv Python package and project manager](https://docs.astral.sh/uv/)
2. At the moment of writing, the raw data is freely available on https://ai.stanford.edu/~amaas/data/sentiment/. Extract the data into the `data/raw/` subdirectory

The subsequent steps are model-specific. The models themselves can be trained using the Python code available in the `src` folder. For example, to execute run the pipeline that estimates the bag of words model, run
```
uv run src/BagOfWords.py
```

These models are registered locally in the SQLite database `mlflow.db`. You can access the logged parameters, metrics, and artifacts through the mlflow server.
```
mlflow server --backend-store-uri sqlite:///mlflow-runs/mlflow.db --host 127.0.0.1 --port 5000
```

## Performance
The table below offers a performance overview. More extensive descriptions of the models can be found in the wiki.
