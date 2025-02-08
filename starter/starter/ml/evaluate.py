'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def evaluate_model_on_slices(model, data, categorical_features, target_column):
    """
    Avalia o desempenho do modelo em fatias de dados categóricos.
    """
    results = {}

    for feature in categorical_features:
        unique_values = data[feature].unique()
        results[feature] = {}

        for value in unique_values:
            slice_data = data[data[feature] == value]
            X_slice = slice_data.drop(columns=[target_column])
            y_slice = slice_data[target_column]

            y_pred = model.predict(X_slice)

            accuracy = accuracy_score(y_slice, y_pred)
            precision = precision_score(y_slice, y_pred, average='weighted')
            recall = recall_score(y_slice, y_pred, average='weighted')
            f1 = f1_score(y_slice, y_pred, average='weighted')

            results[feature][value] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

    return results

if __name__ == "__main__":
    # Dados de exemplo para teste
    data = pd.DataFrame({
        'feature1': ['A', 'B', 'A', 'B', 'A', 'B'],
        'feature2': ['X', 'X', 'Y', 'Y', 'X', 'Y'],
        'target': [0, 1, 0, 1, 0, 1]
    })

    # Modelo de exemplo (usando DummyClassifier para simular um modelo treinado)
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy="stratified")
    model.fit(data[['feature1', 'feature2']], data['target'])

    # Avaliar em fatias
    categorical_features = ['feature1', 'feature2']
    target_column = 'target'
    results = evaluate_model_on_slices(model, data, categorical_features, target_column)

    # Exibir resultados
    for feature, slices in results.items():
        print(f"Feature: {feature}")
        for value, metrics in slices.items():
            print(f"  Value: {value}")
            print(f"    Accuracy: {metrics['accuracy']:.2f}")
            print(f"    Precision: {metrics['precision']:.2f}")
            print(f"    Recall: {metrics['recall']:.2f}")
            print(f"    F1 Score: {metrics['f1_score']:.2f}")
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier

from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
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