import argparse
import numpy as np
import mlflow
import optuna
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sqlite3
import matplotlib.pyplot as plt


def ensure_sqlite_backend(db_path):
    if not os.path.exists(db_path):
        print(f"Creating SQLite database at {db_path}...")
        conn = sqlite3.connect(db_path)
        conn.close()


def main(
    vocabulary_size=1000, mlflow_experiment="default", run_name="default", n_trials=10
) -> None:
    def create_X_y(
        data: pd.DataFrame, target: str = "label", feature: str = "word_sequence"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert tokenized review data into binary feature matrix and label array.

        Transforms word sequences (hyphen-separated token IDs) into a binary
        bag-of-words representation where each element indicates the presence (1)
        or absence (0) of a token in the review. We do not preserve token counts,
        only token presence.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing tokenized reviews and their labels.
        target : str, default="label"
            Name of the column containing the target labels (e.g., sentiment).
        feature : str, default="word_sequence"
            Name of the column containing hyphen-separated token ID sequences
            (e.g., "5-12-45-3").

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            X : np.ndarray of shape (n_samples, vocabulary_size + 1)
                Binary feature matrix where X[i, j] = 1 if token j appears in
                review i, otherwise 0.
            y : np.ndarray of shape (n_samples,)
                Array of target labels.
        """
        # Extract features and labels
        y = data[target].values

        # Convert word_sequence strings to token count matrix X_count
        X_count = np.zeros((len(data), vocabulary_size + 1), dtype=int)
        for i, sequence in enumerate(data[feature]):
            tokens = [int(num) for num in sequence.split("-")]
            for token in tokens:
                if token <= vocabulary_size:
                    X_count[i, token] += 1
                else:
                    raise ValueError(
                        f"Token index {token} exceeds vocabulary size {vocabulary_size}"
                    )

        # Create matrix showing presence (1) or absence (0) of tokens in review
        X = (X_count > 0).astype(int)

        return X, y

    # Set MLflow tracking URI to the mlflow-runs folder
    tracking_uri = os.path.abspath("mlflow-runs")
    db_path = os.path.join(tracking_uri, "mlflow.db")
    artifacts_path = os.path.join(tracking_uri, "artifacts")

    # Ensure SQLite database exists
    ensure_sqlite_backend(db_path)
    os.makedirs(artifacts_path, exist_ok=True)

    # Set MLflow tracking URI with SQLite backend
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    # Set up MLflow experiment with artifact location
    if mlflow_experiment:
        print(f"\nSetting up MLflow experiment: {mlflow_experiment}...")
        experiment = mlflow.get_experiment_by_name(mlflow_experiment)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                mlflow_experiment, artifact_location=f"file://{artifacts_path}"
            )
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        print("\nUsing default MLflow experiment...")
        mlflow.set_experiment("Default")

    # Prepare training data
    train_data = pd.read_csv("data/2_model_input/train/train_reviews_tokenized.csv")
    X_train, y_train = create_X_y(train_data)

    # Prepare validation data
    validation_data = pd.read_csv(
        "data/2_model_input/validation/validation_reviews_tokenized.csv"
    )
    X_val, y_val = create_X_y(validation_data)

    # Hyperparameter optimization with Optuna
    def optuna_objective(trial):
        # Use l1 penalty for sparsity and grid search over C
        C_values = np.logspace(-2, 2, num=10)
        C = trial.suggest_categorical("C", C_values.tolist())

        # Train model with suggested hyperparameters
        model = LogisticRegression(
            max_iter=1000, random_state=42, C=C, l1_ratio=1, solver="liblinear"
        )
        model.fit(X_train, y_train)

        # Evaluate on validation set
        val_accuracy = model.score(X_val, y_val)

        # Log trial to MLflow
        with mlflow.start_run(run_name=f"{run_name}_trial_{trial.number}", nested=True):
            mlflow.log_param("C", C)
            mlflow.log_param("penalty", "l1")
            mlflow.log_param("vocabulary_size", vocabulary_size)
            mlflow.log_metric("validation_accuracy", val_accuracy)

        return val_accuracy

    # Create Optuna study and optimize
    print(f"\nStarting Optuna optimization with {n_trials} trials...")
    C_values = np.logspace(-2, 2, num=10)
    search_space = {"C": C_values.tolist()}
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.GridSampler(search_space)
    )

    # Run MLflow study
    with mlflow.start_run(run_name=run_name):
        # Run Optuna hyperparameter search
        study.optimize(optuna_objective, n_trials=n_trials)

        # Extract trial results as DataFrame
        trials_df = study.trials_dataframe()

        # Create visualization of C vs validation accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        C_values = trials_df["params_C"].values
        val_accuracies = trials_df["value"].values

        # Create scatter plot of C vs validation accuracy
        ax.scatter(C_values, val_accuracies, alpha=0.6, s=100, edgecolors="black")
        ax.set_xscale("log")
        ax.set_xlabel("Regularization Parameter C", fontsize=12)
        ax.set_ylabel("Validation Accuracy", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add best trial marker
        best_trial_idx = trials_df["value"].idxmax()
        best_C = trials_df.loc[best_trial_idx, "params_C"]
        best_acc = trials_df.loc[best_trial_idx, "value"]
        ax.scatter(
            [best_C],
            [best_acc],
            color="red",
            s=200,
            marker="*",
            edgecolors="black",
            label=f"Best Trial (C={best_C:.4f}, Acc={best_acc:.4f})",
            zorder=5,
        )
        ax.legend(fontsize=10)
        plt.tight_layout()

        # Save plot
        plot_path = f"mlflow-runs/optuna_output/{run_name}_C_vs_accuracy.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved visualization to: {plot_path}")
        plt.close()

        # Log artifacts to MLflow
        mlflow.log_artifact(plot_path)

        # Get best parameters
        best_params = study.best_params
        print(f"\nBest parameters: {best_params}")
        print(f"Best validation accuracy: {study.best_value:.4f}")

        # Train final model with best parameters
        best_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=best_params["C"],
            l1_ratio=1,
            solver="liblinear",
        )
        best_model.fit(X_train, y_train)

        # Log best parameters
        mlflow.log_param("best_C", best_params["C"])
        mlflow.log_param("penalty", "l1")
        mlflow.log_param("vocabulary_size", vocabulary_size)
        mlflow.log_param("n_trials", n_trials)

        # Log model to MLflow (and overwrite if existing)
        model_path = os.path.join("mlflow-models", run_name)
        if os.path.exists(model_path):
            import shutil

            shutil.rmtree(model_path)
        os.makedirs(model_path, exist_ok=True)
        mlflow.sklearn.save_model(
            best_model, model_path, conda_env=None, pip_requirements=None
        )

        # Evaluate train and validation accur
        train_accuracy = best_model.score(X_train, y_train)
        val_accuracy = best_model.score(X_val, y_val)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("validation_accuracy", val_accuracy)

        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")

        # Evaluate test data
        test_data = pd.read_csv("data/2_model_input/test/test_reviews_tokenized.csv")
        X_test, y_test = create_X_y(test_data)
        test_accuracy = best_model.score(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print("Model trained successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model training script for BoW classification"
    )
    parser.add_argument(
        "--vocabulary-size", type=int, default=1000, help="Vocabulary size"
    )
    parser.add_argument("--run-name", type=str, default="BoW", help="MLflow run name")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of Optuna trials (only used with --use-optuna)",
    )

    args = parser.parse_args()
    main(
        vocabulary_size=args.vocabulary_size,
        run_name=args.run_name,
        n_trials=args.n_trials,
    )
