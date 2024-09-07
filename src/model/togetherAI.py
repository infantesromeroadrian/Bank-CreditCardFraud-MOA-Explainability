from together import Together
from dotenv import load_dotenv
import os

# Clase de integración con Together AI
class TogetherAIIntegration:
    def __init__(self):
        # Cargar el archivo .env
        load_dotenv()

        # Obtener la API key desde el archivo .env
        self.api_key = os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY no está configurado en el archivo .env")

        # Inicializar el cliente de Together AI con la API key
        self.client = Together(api_key=self.api_key)

        # Modelos a utilizar para la generación de características e interpretación
        self.models = {
            "feature_generation": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "result_interpretation": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        }

    def generate_features(self, data):
        """Genera nuevas características usando LLM."""
        prompt = f"Given the following customer data, suggest 3 new relevant features:\n{data.to_dict()}"
        response = self.client.chat.completions.create(
            model=self.models["feature_generation"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content

    def interpret_results(self, prediction, probabilities, customer_data):
        """Interpreta los resultados de la predicción usando LLM."""
        prompt = f"""
        Given the following prediction and customer data, provide a detailed explanation:
        Prediction: {'Fraud' if prediction == 1 else 'No Fraud'}
        Fraud Probability: {probabilities[1]:.2f}
        Customer Data: {customer_data.to_dict()}
        """
        response = self.client.chat.completions.create(
            model=self.models["result_interpretation"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content
