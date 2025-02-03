# Put the code for your API here.
import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

base_path = os.path.dirname(__file__)  # Diretório onde o main.py está
model_path = os.path.join(base_path, "model")

# Agora você pode carregar os modelos
model = joblib.load(os.path.join(model_path, "random_forest_model.pkl"))
encoder = joblib.load(os.path.join(model_path, "encoder.pkl"))
lb = joblib.load(os.path.join(model_path, "lb.pkl"))


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API de inferência do modelo de renda!"}

class InputData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
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

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
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

    input_data = pd.DataFrame([data.dict()])
    
    cat_features = [
        "workclass", "education", "marital_status", "occupation", "relationship",
        "race", "sex", "native_country"
    ]
    
    from starter.ml.data import process_data
    X, _, _, _ = process_data(input_data, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)
    
    prediction = model.predict(X)
    
    return {"prediction": ">=50K" if prediction[0] == 1 else "<50K"}
