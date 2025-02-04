import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import sys
import os
import joblib
import pickle
from pathlib import Path

from starter.ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def data():
    """Fixture to create data"""
    X_train, y_train = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X_test, y_test = make_classification(n_samples=30, n_features=10, n_classes=2, random_state=42)
    rf_model = train_model(X_train, y_train)
    
    return X_train, y_train, X_test, y_test, rf_model

'''
@pytest.fixture
def model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, '..', 'model')
    model_path = os.path.join(model_dir, "random_forest_model.pkl")
    print("Caminho do modelo:", model_path)
    model = joblib.load(model_path)
    return model
'''

@pytest.fixture
def model():
    # Verifica se está rodando no GitHub Actions
    if "GITHUB_WORKSPACE" in os.environ:
        print("Rodando no GitHub Actions")
        base_dir = Path(os.environ["GITHUB_WORKSPACE"])
    else:
        print("Rodando no ambiente local")
        script_dir = Path(__file__).resolve().parent
        base_dir = script_dir.parent
    model_path = base_dir / "model" / "random_forest_model.pkl"
    print("Caminho do modelo:", model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {model_path}")

    model = joblib.load(model_path)
    return model


def test_train_model(data,model):
    """Testing if model was saved and correcly trained"""
    X_train, y_train, _, _, _ = data
    assert isinstance(model, RandomForestClassifier)

def test_compute_model_metrics(data):
    """Testing if metrics are floats"""
    _, _, X_test, y_test, rf_model = data
    preds = inference(rf_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

def test_inference(data):
    """Testing if inference function return predictions!"""
    _, _, X_test, _, model = data
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)