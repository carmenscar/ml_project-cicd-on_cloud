# Put the code for your API here.
import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference
from pathlib import Path
import uvicorn

logging.basicConfig(filename='logs.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

base_path = Path(__file__).resolve().parent
model_path = base_path / "model" / "random_forest_model.pkl"
encoder_path = base_path / "model" / "encoder.pkl"
lb_path = base_path / "model" / "lb.pkl"

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
lb = joblib.load(lb_path)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API de inferência do modelo de renda!"}

class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class ConfigDict:
        json_schema_extra = {
            "example": {
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
        }


@app.post("/predict/")
def predict(data: InputData):

    input_data = pd.DataFrame([data.model_dump()])
    
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    sample = {key.replace('_', '-'): [value] for key, value in data.__dict__.items()}
    input_data = pd.DataFrame.from_dict(sample)
    logging.info("Processando sample")
    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    output = inference(model=model, X=X)[0]
    str_out = '<=50K' if output == 0 else '>50K'
    logging.info("prediction:", str_out)

    return {"prediction": str_out}



if __name__ == "__main__":
    import uvicorn
    import time
    import requests
    import threading

    def run_api():
        uvicorn.run(app, host="127.0.0.1", port=8000)

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

    print(f"Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")

