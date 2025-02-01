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