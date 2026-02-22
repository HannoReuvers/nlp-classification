import numpy as np

import logging

import mlflow
import optuna
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sqlite3
import matplotlib.pyplot as plt

from src.config import Config

logger = logging.getLogger("BoWModel   ")


class BagOfWordsEstimator:
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_validation: pd.DataFrame,
        df_test: pd.DataFrame,
        mlflow_experiment: str = "default",
        run_name: str = "default",
        n_trials: int = 10,
        config: Config = Config(),
    ):
        self.df_train = df_train
        self.df_validation = df_validation
        self.df_test = df_test
        self.mlflow_experiment = mlflow_experiment
        self.run_name = run_name
        self.n_trials = n_trials
        self.config = config

    def create_X_y(
        self,
        data: pd.DataFrame,
        target: str = "label",
        feature: str = "word_sequence",
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
        X_count = np.zeros((len(data), self.config.VOCABULARY_SIZE + 1), dtype=int)
        for i, sequence in enumerate(data[feature]):
            tokens = [int(num) for num in sequence.split("-")]
            for token in tokens:
                if token <= self.config.VOCABULARY_SIZE:
                    X_count[i, token] += 1
                else:
                    raise ValueError(
                        f"Token index {token} exceeds vocabulary size {self.config.VOCABULARY_SIZE}"
                    )

        # Create matrix showing presence (1) or absence (0) of tokens in review
        X = (X_count > 0).astype(int)

        return X, y

    def estimate_BoW_model(self):  # model: no cover
        # Set MLflow tracking URI to the mlflow-runs folder
        tracking_uri = self.config.MLFLOW_TRACKING_URI
        db_path = tracking_uri / "mlflow.db"
        artifacts_path = tracking_uri / "artifacts"

        # Ensure SQLite database exists
        if not os.path.exists(db_path):
            print(f"Creating SQLite database at {db_path}...")
            conn = sqlite3.connect(db_path)
            conn.close()
        os.makedirs(artifacts_path, exist_ok=True)

        # Set MLflow tracking URI with SQLite backend
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        # Set up MLflow experiment with artifact location
        if self.mlflow_experiment:
            experiment = mlflow.get_experiment_by_name(self.mlflow_experiment)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.mlflow_experiment, artifact_location=f"file://{artifacts_path}"
                )
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_id=experiment_id)
        else:
            logger.info("\nUsing default MLflow experiment...")
            mlflow.set_experiment("Default")

        # Prepare X,y for training and validation
        X_train, y_train = self.create_X_y(self.df_train)
        X_val, y_val = self.create_X_y(self.df_validation)

        # Hyperparameter optimization with Optuna
        def optuna_objective(trial):
            C = trial.suggest_categorical("C", C_values)

            # Train model with suggested hyperparameters
            model = LogisticRegression(
                max_iter=1000, random_state=42, C=C, l1_ratio=1, solver="liblinear"
            )
            model.fit(X_train, y_train)

            # Evaluate on validation set
            val_accuracy = model.score(X_val, y_val)

            # Log trial to MLflow
            with mlflow.start_run(
                run_name=f"{self.run_name}_trial_{trial.number}", nested=True
            ):
                mlflow.log_param("C", C)
                mlflow.log_param("penalty", "l1")
                mlflow.log_param("vocabulary_size", self.config.VOCABULARY_SIZE)
                mlflow.log_metric("validation_accuracy", val_accuracy)

            return val_accuracy

        # Create Optuna study and optimize
        C_values = np.logspace(-1, 1, num=10)
        search_space = {"C": C_values.tolist()}

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.GridSampler(search_space)
        )

        # Run MLflow study
        with mlflow.start_run(run_name=self.run_name):
            # Run Optuna hyperparameter search
            study.optimize(optuna_objective, n_trials=len(C_values))

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
            plot_path = f"mlflow-runs/optuna_output/{self.run_name}_C_vs_accuracy.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()

            # Log artifacts to MLflow
            mlflow.log_artifact(plot_path)

            # Get best parameters
            best_params = study.best_params

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
            mlflow.log_param("vocabulary_size", self.config.VOCABULARY_SIZE)
            mlflow.log_param("n_trials", self.n_trials)

            # Log model to MLflow (and overwrite if existing)
            model_path = os.path.join("mlflow-models", self.run_name)
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

            logger.info(f"Training accuracy: {train_accuracy:.4f}")
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")

            # Evaluate test data
            X_test, y_test = self.create_X_y(self.df_test)
            test_accuracy = best_model.score(X_test, y_test)
            mlflow.log_metric("test_accuracy", test_accuracy)
            logger.info(f"Test accuracy: {test_accuracy:.4f}")
            logger.info("Model trained successfully!")
