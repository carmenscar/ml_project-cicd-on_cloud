import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
from starter.ml.data import process_data
from starter.ml.model import inference, compute_model_metrics

def evaluate_on_slices(model, data, categorical_features, target, encoder, lb):
    """
    Avalia o desempenho do modelo em diferentes fatias dos dados e salva os resultados em um arquivo.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Modelo treinado.
    data : pd.DataFrame
        Dados completos, incluindo features e target.
    categorical_features : list[str]
        Lista contendo os nomes das features categóricas.
    target : str
        Nome da coluna target.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Encoder treinado para as features categóricas.
    lb : sklearn.preprocessing._label.LabelBinarizer
        LabelBinarizer treinado para o target.
    output_file : str, optional
        Caminho do arquivo onde os resultados serão salvos. Se None, os resultados não são salvos.

    Returns
    -------
    results : dict
        Dicionário contendo as métricas de avaliação (precision, recall, fbeta) para cada fatia.
    """
    results = {}

    # Processa os dados completos
    X, y, _, _ = process_data(data, categorical_features=categorical_features, label=target, training=False, encoder=encoder, lb=lb)

    # Avalia o modelo em cada fatia das features categóricas
    for feature in categorical_features:
        results[feature] = {}
        unique_values = data[feature].unique()

        for value in unique_values:
            # Filtra os dados para a fatia atual
            slice_mask = data[feature] == value
            X_slice = X[slice_mask]
            y_slice = y[slice_mask]

            # Faz as previsões e calcula as métricas
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)

            # Armazena os resultados
            results[feature][value] = {
                'precision': precision,
                'recall': recall,
                'fbeta': fbeta
            }

        with open("slice_output.txt", 'w') as f:
            for feature, values in results.items():
                f.write(f"Feature: {feature}\n")
                for value, metrics in values.items():
                    f.write(f"  Value: {value}\n")
                    f.write(f"    Precision: {metrics['precision']:.4f}\n")
                    f.write(f"    Recall: {metrics['recall']:.4f}\n")
                    f.write(f"    F-beta: {metrics['fbeta']:.4f}\n")
                f.write("\n")

    return results