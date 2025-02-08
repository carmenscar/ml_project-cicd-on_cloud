from fastapi.testclient import TestClient

import sys
import os
import uvicorn
import time
import requests
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app

client = TestClient(app)

def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000)

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bem-vindo à API de inferência do modelo de renda!"}

def test_post_prediction_1():
    thread = threading.Thread(target=run_api, daemon=True)
    thread.start()

    # Aguarde a API iniciar
    time.sleep(2)

    # Testa um POST automaticamente
    url = "http://127.0.0.1:8000/predict/"
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    response = requests.post(url, json=payload)
    expected_values = {"<=50K", ">50K"}
    #assert response.status_code == 200, f"Status code inesperado: {response.status_code}"
    #assert response.json()["prediction"] in expected_values, f"Resposta inesperada: {response.json()}"
    data = response.json()
    assert response.status_code == 200  
    assert data["prediction"] in expected_values
'''
def test_post_prediction_2():
    response = client.post("/predict/", json={
        "age": 50,
        "workclass": "Private",
        "fnlgt": 234721,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States"
    })
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<50K", ">=50K"]
'''