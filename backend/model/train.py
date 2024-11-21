import os
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import javalang
from joblib import dump
import time
from datetime import datetime
import json

def split_by_cases(dataset_dir):
    """
    Divide los casos por carpetas completas en lugar de archivos individuales
    """
    cases = list(dataset_dir.iterdir())
    case_ids = list(set([c.parent for c in cases]))
    train_cases, test_cases = train_test_split(
        case_ids, 
        test_size=0.2, 
        random_state=42
    )
    return train_cases, test_cases

def extract_features(original_file, comparison_file, metrics_collector=None):
    """
    Extrae características más completas para cada comparación
    """
    features = []
    
    # Token overlap
    token_overlap = calculate_token_overlap(original_file, comparison_file, metrics_collector)
    features.append(token_overlap)
    
    # AST similarity
    ast_similarity = calculate_ast_similarity(original_file, comparison_file, metrics_collector)
    features.append(ast_similarity)
    
    # Podrías añadir más características aquí
    
    return features

def evaluate_model(y_true, y_pred):
    """
    Evalúa el modelo usando múltiples umbrales
    """
    metrics = {}
    for threshold in [30, 40, 50, 60, 70]:
        predictions = y_pred > threshold
        true_labels = y_true > threshold
        metrics[f'threshold_{threshold}'] = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions),
            'f1': f1_score(true_labels, predictions)
        }
    return metrics

def train_with_cv(X, y, model):
    """
    Entrena el modelo usando validación cruzada
    """
    cv_scores = cross_validate(
        model, 
        X, 
        y,
        cv=5,
        scoring=['accuracy', 'precision', 'recall', 'f1']
    )
    return cv_scores

def save_metrics(metrics, model_dir):
    """
    Guarda métricas más detalladas incluyendo desviación estándar
    """
    metrics_data = {
        "model_performance": {
            "mae": float(metrics["mae"]),
            "thresholds": metrics["thresholds"],
            "cross_validation": {
                "accuracy": {
                    "mean": float(np.mean(metrics["test_accuracy"])),
                    "std": float(np.std(metrics["test_accuracy"]))
                },
                "precision": {
                    "mean": float(np.mean(metrics["test_precision"])),
                    "std": float(np.std(metrics["test_precision"]))
                },
                "recall": {
                    "mean": float(np.mean(metrics["test_recall"])),
                    "std": float(np.std(metrics["test_recall"]))
                },
                "f1_score": {
                    "mean": float(np.mean(metrics["test_f1"])),
                    "std": float(np.std(metrics["test_f1"]))
                }
            }
        },
        "training_info": {
            "total_samples": len(metrics["total_samples"]),
            "training_samples": len(metrics["training_samples"]),
            "test_samples": len(metrics["test_samples"]),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    try:
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        metrics_path = model_dir / "metrics.json"
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
        return True
    except Exception as e:
        print(f"Error al guardar métricas: {e}")
        return False

def main():
    try:
        print("Iniciando entrenamiento del modelo mejorado...")
        
        # Configurar directorios
        dataset_dir = get_dataset_path()
        model_dir = Path('model')
        model_dir.mkdir(exist_ok=True)
        
        # Dividir casos por carpetas
        train_cases, test_cases = split_by_cases(dataset_dir)
        
        features = []
        labels = []
        processed_files = set()
        
        # Procesar casos de entrenamiento
        for case in train_cases:
            original_file = load_original_file(case)
            plagiarized_files = load_plagiarized_files(case)
            non_plagiarized_files = load_non_plagiarized_files(case)
            
            for file in plagiarized_files + non_plagiarized_files:
                if str(file) not in processed_files:
                    processed_files.add(str(file))
                    features.append(extract_features(original_file, file))
                    labels.append(get_label(original_file, file))
        
        X = np.array(features)
        y = np.array(labels)
        
        # Entrenar modelo con validación cruzada
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        cv_scores = train_with_cv(X, y, model)
        
        # Entrenar modelo final
        model.fit(X, y)
        
        # Evaluar en conjunto de prueba
        test_features = []
        test_labels = []
        
        for case in test_cases:
            original_file = load_original_file(case)
            for file in load_plagiarized_files(case) + load_non_plagiarized_files(case):
                test_features.append(extract_features(original_file, file))
                test_labels.append(get_label(original_file, file))
        
        X_test = np.array(test_features)
        y_test = np.array(test_labels)
        
        y_pred = model.predict(X_test)
        
        # Calcular métricas completas
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "thresholds": evaluate_model(y_test, y_pred),
            "test_accuracy": cv_scores['test_accuracy'],
            "test_precision": cv_scores['test_precision'],
            "test_recall": cv_scores['test_recall'],
            "test_f1": cv_scores['test_f1'],
            "total_samples": y,
            "training_samples": y,
            "test_samples": y_test
        }
        
        # Guardar modelo y métricas
        dump(model, model_dir / 'classifier.joblib')
        save_metrics(metrics, model_dir)
        
        print("\nMétricas del modelo:")
        print(f"MAE: {metrics['mae']:.4f}")
        print("\nResultados por umbral:")
        for threshold, scores in metrics['thresholds'].items():
            print(f"\n{threshold}:")
            for metric, value in scores.items():
                print(f"{metric}: {value:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"Error general: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
