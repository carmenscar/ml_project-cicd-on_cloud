import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    model_path = base_dir / "model" / "random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(script_dir)
    print(base_dir)
    print(model_path)
    
    return model



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : random forest
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

if __name__ == "__main__":
    X_train = np.random.rand(100, 5)  
    y_train = np.random.randint(0, 2, 100)  
    model = train_model(X_train, y_train)
    X_test = np.random.rand(10, 5) 
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_train[:10], preds) 
    print(f"Precision: {precision}, Recall: {recall}, F-beta: {fbeta}")

