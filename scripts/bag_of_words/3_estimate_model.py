import argparse
import numpy as np
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression


def main(vocabulary_size=1000, mlflow_experiment=None, run_name="default") -> None:
    # Set up MLflow experiment
    if mlflow_experiment:
        print(f"\nSetting up MLflow experiment: {mlflow_experiment}...")
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run(run_name=run_name)

    # Read the training data
    data = pd.read_csv("data/2_model_input/train/train_reviews_tokenized.csv")

    # Columns of interest
    target_column = "label"
    feature_column = "word_sequence"

    # Extract features and labels
    y_train = data[target_column].values

    # Convert word_sequence strings to token count matrix X_count
    X_train_count = np.zeros((len(data), vocabulary_size + 1), dtype=int)
    for i, sequence in enumerate(data[feature_column]):
        tokens = [int(num) for num in sequence.split("-")]
        for token in tokens:
            if token <= vocabulary_size:
                X_train_count[i, token] += 1
            else:
                raise ValueError(
                    f"Token index {token} exceeds vocabulary size {vocabulary_size}"
                )

    # Create matrix showing presence (1) or absence (0) of tokens in review
    X_train = (X_train_count > 0).astype(int)

    # Fit logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    print("Model trained successfully!")
    print(f"Training accuracy: {model.score(X_train, y_train):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model training script for BoW classification"
    )
    parser.add_argument(
        "--vocabulary-size", type=int, default=1000, help="Vocabulary size"
    )
    parser.add_argument(
        "--mlflow-experiment", type=str, default=None, help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name", type=str, default="default", help="MLflow run name"
    )

    args = parser.parse_args()
    main(
        vocabulary_size=args.vocabulary_size,
        mlflow_experiment=args.mlflow_experiment,
        run_name=args.run_name,
    )
