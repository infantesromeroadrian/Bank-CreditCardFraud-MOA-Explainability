# Bank Credit Card Fraud Detection with MOA and Explainability 🔍💳

This project focuses on detecting credit card fraud using a Mixture of Agents (MOA) model, combining machine learning with explainability features provided by a Large Language Model (LLM). The goal is to enhance transparency and provide understandable insights into fraud detection results.

## 🚀 Features
- **Fraud Detection Model**: A MOA model trained to identify fraudulent transactions.
- **Data Preparation**: Preprocessing pipeline that normalizes and handles missing values in the data.
- **Explainability**: Integration with an LLM to generate clear, human-readable explanations for each prediction, ensuring transparency in decision-making.
- **Streamlit Interface**: User-friendly interface to input transaction data and view predictions with detailed explanations.

## 📂 Project Structure
```
├── README.md
├── assets
├── bankcreditcardfraud
│   └── __init__.py
├── data
│   ├── processed_data
│   ├── raw_data
│   ├── test
│   ├── train
│   └── val
├── docs
├── models
│   ├── data_preparation.pkl
│   └── moa_model
├── notebooks
├── poetry.lock
├── pyproject.toml
├── src
│   ├── features
│   ├── main.py
│   ├── model
│   └── utils
└── tests
```

## 🛠 Installation & Usage
Clone the Repository:


```bash
git clone https://github.com/yourusername/Bank-CreditCard-MOA-Explicability.git
cd Bank-CreditCard-MOA-Explicability
```

Set up the Environment:


Install the required Python dependencies with poetry or pip.

```bash
poetry install
```

Prepare the Dataset:

Place your dataset (creditcard.csv) in the data/raw_data/ directory.
Run the Streamlit App:

```bash
streamlit run app.py
```

Environment Variables:

Add your API key for the LLM in a .env file:

```bash
TOGETHER_API_KEY=your_api_key
```

📊 Model and Data Preprocessing
The project uses StandardScaler for data normalization and SMOTE to balance the dataset.
The model is trained to classify transactions as fraudulent or not, with detailed explanations generated for each result.

🧠 Explainability
Using the TogetherAI LLM, this project adds human-readable insights into each prediction, helping users understand why a transaction is classified as fraudulent or legitimate.


🤖 Technology Stack
Python 🐍
scikit-learn for machine learning
MLflow for model management
TogetherAI for explainability
Streamlit for the user interface
Poetry for dependency management

🌟 Contributing
Feel free to open issues or pull requests to enhance the project.

⚖️ License
This project is licensed under the MIT License. See the LICENSE file for more information.


