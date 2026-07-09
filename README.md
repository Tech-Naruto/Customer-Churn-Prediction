# Customer Churn Prediction Project

## 1. Project Overview
A deep learning project aimed at predicting customer churn for a business using historical customer data. It analyzes user demographics, account details, and transaction histories to classify whether a customer is likely to stay or leave (churn). 

## 2. Project Directory Structure

    ├── artifacts/
    │   ├── model.pkl                   # Trained serialized ANN weights/model
    │   └── preprocessor.pkl            # Scalers/Encoders for data preprocessing
    ├── dataset/
    │   └── Churn_Modelling.csv         # Raw customer historical data
    ├── notebooks/
    │   ├── ANN Classification.ipynb    # Model training using Keras with PyTorch backend
    │   └── prediction.ipynb            # Model inference pipeline testing
    ├── .gitignore                      # Files to exclude from Git tracking
    ├── app.py                          # UI/Deployment application file
    └── README.md                       # Project report 

## 3. Dataset Characteristics
* Source: Bank Customer Churn Dataset (Churn_Modelling.csv)
* Features: Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, and Estimated Salary.
* Target Variable: Exited (Binary)
  * 0: Retained (Customer stays)
  * 1: Churned (Customer leaves)

## 4. Methodology & Model Architecture
* Framework: Keras utilizing a unified PyTorch (torch) backend.
* Algorithm Used: Artificial Neural Network (ANN)
  * Input layer mapping the processed numerical and categorical inputs.
  * Hidden layers optimized with ReLU activation functions and Dropout layers to prevent overfitting.
  * Output layer with a Sigmoid activation function for binary classification probability.
* Preprocessing Pipeline: Saved inside preprocessor.pkl
  * One-Hot Encoding: Applied to categorical feature (Geography).
  * Ordinal Encoding: Applied to categorical feature (Gender).
  * Standard Scaling: Applied to numerical features (Age, Balance, CreditScore, etc.) to stabilize neural network weight updates.

## 5. How to Run Locally

### Prerequisites
Make sure you have Miniconda/Anaconda installed on your machine.

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <your-github-repo-link>
   cd "Churn Prediction"

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the application:
   ```bash
   streamlit run app.py