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

logging.basicConfig(filename='logs.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

base_path = os.path.dirname(os.path.abspath(__file__))
#model = joblib.load(os.path.join(base_path,"model", "random_forest_model.pkl"))
#encoder = joblib.load(os.path.join(base_path,"model","encoder.pkl"))
#lb = joblib.load(os.path.join(base_path, "model", "lb.pkl"))

if os.path.exists(os.path.join(base_path,"model", "random_forest_model.pkl")):
    model = joblib.load(os.path.join(base_path,"model", "random_forest_model.pkl"))
else:
    logging.error("Erro: Arquivo de model não encontrado!")

if os.path.exists(os.path.join(base_path,"model","encoder.pkl")):
    encoder = joblib.load(os.path.join(base_path,"model","encoder.pkl"))
else:
    logging.error("Erro: Arquivo de encoder não encontrado!")

if os.path.exists(os.path.join(base_path, "model", "lb.pkl")):
    lb = joblib.load(os.path.join(base_path, "model", "lb.pkl"))
else:
    logging.error("Erro: Arquivo de lb não encontrado!")

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
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    X, _, _, _ = process_data(input_data, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb)
    print(X.shape)
    prediction = model.predict(X)
    
    return {"prediction": ">=50K" if prediction[0] == 1 else "<50K"}
