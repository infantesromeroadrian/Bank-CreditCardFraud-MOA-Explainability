from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import numpy as np
import os
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import logging
from src.utils.decorators import timer_decorator, error_handler, log_decorator

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import mlflow
import numpy as np
import os

logger = logging.getLogger(__name__)

class MixtureOfAgents(BaseEstimator, ClassifierMixin):
    def __init__(self, models=None, weights=None, experiment_name="MOA_Experiment", data_dir="../data"):
        self.models = models or []
        self.weights = weights
        self.classes_ = None
        self.experiment_name = experiment_name
        self.data_dir = data_dir
        mlflow.set_experiment(experiment_name)

    @timer_decorator
    @error_handler
    @log_decorator
    def add_model(self, model, weight=1.0):
        """Añade un modelo al conjunto."""
        self.models.append(model)
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)

    @timer_decorator
    @error_handler
    @log_decorator
    def fit(self, X, y):
        """Entrena todos los modelos en el conjunto."""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        with mlflow.start_run(run_name="MOA_Training"):
            for i, model in enumerate(self.models):
                logger.info(f"Entrenando modelo {i+1}/{len(self.models)}: {type(model).__name__}")
                with mlflow.start_run(run_name=f"Model_{i}_Training", nested=True):
                    model.fit(X, y)
                    mlflow.log_param(f"model_{i}", type(model).__name__)
                    mlflow.log_param(f"weight_{i}", self.weights[i])
            logger.info("Entrenamiento completado para todos los modelos")

        return self

    @timer_decorator
    @error_handler
    @log_decorator
    def predict_proba(self, X):
        """Predice las probabilidades de clase para X."""
        check_is_fitted(self)
        X = np.array(X)

        predictions = []
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)
            else:
                pred = np.eye(len(self.classes_))[model.predict(X)]
            predictions.append(weight * pred)

        return np.sum(predictions, axis=0) / np.sum(self.weights)

    @timer_decorator
    @error_handler
    @log_decorator
    def predict(self, X):
        """Predice las etiquetas de clase para X."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @timer_decorator
    @error_handler
    @log_decorator
    def evaluate(self, X, y):
        """Evalúa el rendimiento del modelo."""
        y_pred = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
        return metrics