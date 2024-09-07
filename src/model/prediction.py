import pandas as pd
import mlflow
import mlflow.pyfunc
import joblib
import numpy as np
import os
from together import Together
from dotenv import load_dotenv

# Clase de predicción y preparación de datos
class Prediction:
    def __init__(self, model_path, data_prep_path):
        self.model = self.load_model(model_path)
        self.data_prep = self.load_data_prep(data_prep_path)
        self.expected_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                                 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

    @staticmethod
    def load_model(model_path):
        """Carga el modelo desde el archivo."""
        return mlflow.pyfunc.load_model(model_path)

    @staticmethod
    def load_data_prep(data_prep_path):
        """Carga la instancia de preparación de datos."""
        return joblib.load(data_prep_path)

    def prepare_data(self, data):
        """Prepara los datos para la predicción."""
        # Crear una nueva instancia de la clase de preparación de datos
        data_prep_instance = self.data_prep.__class__(data)

        # Asegúrate de que las columnas esperadas están presentes antes de procesar
        missing_columns = [col for col in self.expected_columns if col not in data.columns]
        if missing_columns:
            raise KeyError(f"Faltan columnas en los datos: {missing_columns}")

        # Realizar la preparación de los datos
        prepared_data = data_prep_instance.prepare_data().get_prepared_data()

        return prepared_data[self.expected_columns]

    def predict(self, data):
        """Realiza predicciones usando el modelo."""
        prepared_data = self.prepare_data(data)
        return self.model.predict(prepared_data)

    def predict_proba(self, data):
        """Realiza predicciones de probabilidad usando el modelo."""
        prepared_data = self.prepare_data(data)
        proba = self.model.predict(prepared_data)
        # Asegurarse de que proba sea un array 2D
        if proba.ndim == 1:
            proba = np.column_stack((1 - proba, proba))
        return proba

    def predict_and_explain(self, data):
        """Realiza predicciones y proporciona explicaciones básicas."""
        predictions = self.predict(data)
        probabilities = self.predict_proba(data)

        results = []
        for i, pred in enumerate(predictions):
            explanation = f"Predicción: {'Fraudulenta' if pred == 1 else 'No Fraudulenta'}"
            explanation += f"\nProbabilidad de Fraude: {probabilities[i][1]:.2f}"
            explanation += f"\nProbabilidad de No Fraude: {probabilities[i][0]:.2f}"
            results.append(explanation)

        return results