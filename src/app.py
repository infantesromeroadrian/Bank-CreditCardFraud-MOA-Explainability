import streamlit as st
import pandas as pd
import joblib
import mlflow.pyfunc
from model.togetherAI import TogetherAIIntegration
from dotenv import load_dotenv
from features.data_preparation import DataPreparation
from model.prediction import Prediction
from utils.decorators import timer_decorator, error_handler, log_decorator

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# Cargar las variables de entorno del archivo .env
load_dotenv()

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model("models/moa_model")

# Cargar el pipeline de preparación de datos
@st.cache_resource
def load_data_prep():
    return joblib.load("/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/Portfolio/BankProjects/BankCreditCardFraud/models/data_preparation.pkl")

# Preprocesar los datos de entrada del cliente
def preprocess_input(data):
    # Cargar el pipeline de preparación
    data_prep = load_data_prep()
    # Preparar los datos utilizando el pipeline de preparación
    prepared_data = data_prep.prepare_data().get_prepared_data()
    return prepared_data

# Interfaz de usuario en Streamlit
def main():
    st.title("Detección de Fraude con Tarjetas de Crédito")

    st.sidebar.header("Datos del Cliente")

    # Sliders para las características de entrada
    V1 = st.sidebar.slider("V1", -5.0, 5.0, 0.0)
    V2 = st.sidebar.slider("V2", -5.0, 5.0, 0.0)
    V3 = st.sidebar.slider("V3", -5.0, 5.0, 0.0)
    V4 = st.sidebar.slider("V4", -5.0, 5.0, 0.0)
    V5 = st.sidebar.slider("V5", -5.0, 5.0, 0.0)
    V6 = st.sidebar.slider("V6", -5.0, 5.0, 0.0)
    V7 = st.sidebar.slider("V7", -5.0, 5.0, 0.0)
    V8 = st.sidebar.slider("V8", -5.0, 5.0, 0.0)
    V9 = st.sidebar.slider("V9", -5.0, 5.0, 0.0)
    V10 = st.sidebar.slider("V10", -5.0, 5.0, 0.0)
    V11 = st.sidebar.slider("V11", -5.0, 5.0, 0.0)
    V12 = st.sidebar.slider("V12", -5.0, 5.0, 0.0)
    V13 = st.sidebar.slider("V13", -5.0, 5.0, 0.0)
    V14 = st.sidebar.slider("V14", -5.0, 5.0, 0.0)
    V15 = st.sidebar.slider("V15", -5.0, 5.0, 0.0)
    V16 = st.sidebar.slider("V16", -5.0, 5.0, 0.0)
    V17 = st.sidebar.slider("V17", -5.0, 5.0, 0.0)
    V18 = st.sidebar.slider("V18", -5.0, 5.0, 0.0)
    V19 = st.sidebar.slider("V19", -5.0, 5.0, 0.0)
    V20 = st.sidebar.slider("V20", -5.0, 5.0, 0.0)
    V21 = st.sidebar.slider("V21", -5.0, 5.0, 0.0)
    V22 = st.sidebar.slider("V22", -5.0, 5.0, 0.0)
    V23 = st.sidebar.slider("V23", -5.0, 5.0, 0.0)
    V24 = st.sidebar.slider("V24", -5.0, 5.0, 0.0)
    V25 = st.sidebar.slider("V25", -5.0, 5.0, 0.0)
    V26 = st.sidebar.slider("V26", -5.0, 5.0, 0.0)
    V27 = st.sidebar.slider("V27", -5.0, 5.0, 0.0)
    V28 = st.sidebar.slider("V28", -5.0, 5.0, 0.0)
    Amount = st.sidebar.slider("Amount", 0.0, 5000.0, 150.0)

    # Crear DataFrame con los datos del cliente
    customer_data = {
        'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6, 'V7': V7, 'V8': V8, 'V9': V9, 'V10': V10,
        'V11': V11, 'V12': V12, 'V13': V13, 'V14': V14, 'V15': V15, 'V16': V16, 'V17': V17, 'V18': V18, 'V19': V19,
        'V20': V20, 'V21': V21, 'V22': V22, 'V23': V23, 'V24': V24, 'V25': V25, 'V26': V26, 'V27': V27, 'V28': V28,
        'Amount': Amount
    }
    customer_df = pd.DataFrame([customer_data])

    st.subheader("Datos del Cliente")
    st.write(customer_df)

    # Cargar el modelo
    model = load_model()

    # Preprocesar los datos
    prepared_data = preprocess_input(customer_df)

    # Asegurarse de que la columna 'Amount' no esté incluida si el modelo no la utilizó durante el entrenamiento
    prepared_data = prepared_data.drop(columns=['Amount'], errors='ignore')

    # Realizar la predicción
    if st.button("Predecir"):
        prediction = model.predict(prepared_data)

        # Asignar probabilidades de forma manual
        probabilities = [0.9, 0.1] if prediction[0] == 1 else [0.1, 0.9]

        # Mostrar el resultado de la predicción
        st.subheader("Resultado de la Predicción")
        if prediction[0] == 1:
            st.error("Posible Fraude Detectado.")
        else:
            st.success("No se detectó Fraude.")

        # Generar explicación con Together AI
        st.subheader("Explicación Detallada")
        together_ai = TogetherAIIntegration()
        explanation = together_ai.interpret_results(prediction[0], probabilities, customer_df.iloc[0])
        st.write(explanation)


if __name__ == "__main__":
    main()
