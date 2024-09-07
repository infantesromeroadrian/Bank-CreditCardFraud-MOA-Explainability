from .prediction import Prediction
from src.model.togetherAI import TogetherAIIntegration


# Integración con la clase Prediction
class EnhancedPrediction(Prediction):
    def __init__(self, model_path, data_prep_path):
        super().__init__(model_path, data_prep_path)
        self.together_ai = TogetherAIIntegration()

    def predict_and_explain(self, data):
        predictions = self.predict(data)
        probabilities = self.predict_proba(data)

        results = []
        for i, pred in enumerate(predictions):
            basic_explanation = f"Predicción: {'Fraudulenta' if pred == 1 else 'No Fraudulenta'}"
            basic_explanation += f"\nProbabilidad de Fraude: {probabilities[i][1]:.2f}"
            basic_explanation += f"\nProbabilidad de No Fraude: {probabilities[i][0]:.2f}"

            # Generar explicación detallada con el modelo LLM
            detailed_explanation = self.together_ai.interpret_results(pred, probabilities[i], data.iloc[i])

            results.append(f"{basic_explanation}\n\nExplicación detallada:\n{detailed_explanation}")

        return results