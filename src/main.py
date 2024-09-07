from features.data_preparation import DataPreparation
from src.model.mixture_of_agents import MixtureOfAgents
from model.prediction import Prediction
from utils.logging_setup import setup_logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
import pickle

logger = setup_logging()


def main():
    # Cargar y preparar los datos
    bank_data = pd.read_csv("../data/raw_data/creditcard.csv")
    data_prep = DataPreparation(bank_data)
    prepared_data = data_prep.prepare_data().get_prepared_data()

    # Entrenar el modelo
    moa = MixtureOfAgents(experiment_name="CreditCardFraudDetection")
    moa.add_model(LogisticRegression(), weight=1.0)
    moa.add_model(DecisionTreeClassifier(), weight=0.8)
    moa.add_model(RandomForestClassifier(), weight=1.2)

    X = prepared_data.drop('Class', axis=1)
    y = prepared_data['Class']
    moa.fit(X, y)

    # Guardar el modelo usando mlflow
    moa_path = "../models/moa_model"
    data_prep_path = "../models/data_preparation.pkl"

    # Guardar el modelo MixtureOfAgents con mlflow
    mlflow.sklearn.save_model(moa, moa_path)

    # Guardar el objeto data_prep
    with open(data_prep_path, "wb") as f:
        pickle.dump(data_prep, f)

    # Datos de ejemplo del nuevo cliente
    new_customer = {
        'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155, 'V5': -0.338321,
        'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698, 'V9': 0.363787, 'V10': 0.090794,
        'V11': -0.551600, 'V12': -0.617801, 'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177,
        'V16': -0.470401, 'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
        'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928, 'V25': 0.128539,
        'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053, 'Amount': 149.62
    }

    # Convertir los datos del nuevo cliente a un DataFrame
    new_customer_df = pd.DataFrame([new_customer])

    # Crear una instancia de Prediction
    predictor = Prediction(moa_path, data_prep_path)

    # Realizar la predicción para el nuevo cliente
    prediction = predictor.predict(new_customer_df)
    print(f"Predicción para el nuevo cliente: {'Fraudulenta' if prediction[0] == 1 else 'No Fraudulenta'}")


if __name__ == "__main__":
    main()
