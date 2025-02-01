import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import sys
import os
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from starter.ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def data():
    """Fixture to create data"""
    X_train, y_train = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X_test, y_test = make_classification(n_samples=30, n_features=10, n_classes=2, random_state=42)
    model = train_model(X_train, y_train)
    
    return X_train, y_train, X_test, y_test, model

def test_train_model(data):
    """Testing if model was saved and correcly trained"""
    X_train, y_train, _, _, _ = data
    model_path = "/home/carmenscar/nd0821-c3-starter-code/starter/model/random_forest_model.pkl"
    model = joblib.load(model_path)
    
    assert isinstance(model, RandomForestClassifier)

def test_compute_model_metrics(data):
    """Testing if metrics are floats"""
    _, _, X_test, y_test, model = data
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

def test_inference(data):
    """Testing if inference function return predictions"""
    _, _, X_test, _, model = data
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)