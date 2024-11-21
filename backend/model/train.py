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

class MetricsCollector:
    def __init__(self):
        self.case_metrics = {}
        self.execution_times = []
        self.token_similarities = []
        self.ast_similarities = []
        self.start_time = None
        
    def start_timing(self):
        self.start_time = time.time()
        
    def stop_timing(self):
        if self.start_time:
            execution_time = time.time() - self.start_time
            self.execution_times.append(execution_time)
            self.start_time = None
            return execution_time
        return 0

def get_dataset_path():
    """
    Obtiene la ruta del dataset considerando diferentes posibles ubicaciones
    """
    # Intentar obtener la ruta del dataset desde la variable de entorno
    dataset_path = os.getenv('DATASET_PATH')
    if dataset_path:
        # Convertir a Path y hacer la ruta absoluta
        dataset_path = Path(dataset_path).absolute()
        print(f"Using dataset path from env: {dataset_path}")
        return dataset_path
    
    # Si no está en la variable de entorno, intentar encontrarlo relativamente
    current_dir = Path(__file__).parent.absolute()
    possible_paths = [
        current_dir.parent / "data" / "IR-Plag-Dataset",
        current_dir.parent.parent / "data" / "IR-Plag-Dataset",
        Path("data/IR-Plag-Dataset"),
        Path("../data/IR-Plag-Dataset"),
        Path("../../data/IR-Plag-Dataset"),
        current_dir / "data" / "IR-Plag-Dataset"
    ]
    
    print("Searching for dataset in possible locations:")
    for path in possible_paths:
        abs_path = path.absolute()
        print(f"Checking {abs_path}")
        if abs_path.exists():
            print(f"Found dataset at: {abs_path}")
            return abs_path
    
    print("Dataset not found in any of the expected locations")
    print("Current working directory:", os.getcwd())
    print("Directory contents:", os.listdir())
    
    raise FileNotFoundError("No se pudo encontrar el directorio del dataset")

def load_dataset(path):
    """
    Carga los casos del dataset desde el directorio especificado
    """
    try:
        if not path.exists():
            raise FileNotFoundError(f"El directorio {path} no existe")
            
        full_paths = list(path.iterdir())
        print(f"Archivos encontrados en {path}:")
        for p in full_paths:
            print(f"  - {p}")
        return full_paths
    except Exception as e:
        print(f"Error al cargar el dataset desde {path}: {e}")
        raise

def load_original_file(path):
    """
    Carga el archivo original de un caso
    """
    try:
        original_path = path / "original"
        if not original_path.exists():
            raise FileNotFoundError(f"No se encontró el directorio 'original' en {path}")
            
        files = list(original_path.iterdir())
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos en {original_path}")
            
        return files[0]
    except Exception as e:
        print(f"Error al cargar archivo original: {e}")
        raise

def load_plagiarized_files(path):
    """
    Carga los archivos plagiados de un caso
    """
    try:
        plagiarized_path = path / "plagiarized"
        plagiarized_files = []
        
        if not plagiarized_path.exists():
            return []
            
        for level_dir in plagiarized_path.iterdir():
            if level_dir.is_dir():
                for subdir in level_dir.iterdir():
                    if subdir.is_dir():
                        plagiarized_files.extend(list(subdir.iterdir()))
                        
        return plagiarized_files
    except Exception as e:
        print(f"Error al cargar archivos plagiados: {e}")
        return []

def load_non_plagiarized_files(path):
    """
    Carga los archivos no plagiados de un caso
    """
    try:
        non_plagiarized_path = path / "non-plagiarized"
        non_plagiarized_files = []
        
        if not non_plagiarized_path.exists():
            return []
            
        for subdir in non_plagiarized_path.iterdir():
            if subdir.is_dir():
                non_plagiarized_files.extend(list(subdir.iterdir()))
                
        return non_plagiarized_files
    except Exception as e:
        print(f"Error al cargar archivos no plagiados: {e}")
        return []

def tokenize(file_path):
    """
    Tokeniza un archivo Java
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()
        tokens = list(javalang.tokenizer.tokenize(code))
        return [token.value for token in tokens]
    except Exception as e:
        print(f"Error al tokenizar {file_path}: {e}")
        return []

def calculate_token_overlap(file1, file2, metrics_collector=None):
    """
    Calcula la superposición de tokens entre dos archivos
    """
    if metrics_collector:
        metrics_collector.start_timing()
    
    try:
        tokens1 = set(tokenize(file1))
        tokens2 = set(tokenize(file2))
        
        if not tokens1 or not tokens2:
            return 0.0
            
        overlap = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        
        if metrics_collector:
            metrics_collector.token_similarities.append(overlap)
            metrics_collector.stop_timing()
            
        return overlap
    except Exception as e:
        print(f"Error al calcular superposición de tokens: {e}")
        return 0.0

def calculate_ast_similarity(file1, file2, metrics_collector=None):
    """
    Calcula la similitud AST entre dos archivos
    """
    if metrics_collector:
        metrics_collector.start_timing()
        
    try:
        def parse_ast(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                code = file.read()
            tree = javalang.parse.parse(code)
            return [node.__class__.__name__ for _, node in tree]

        nodes1 = parse_ast(file1)
        nodes2 = parse_ast(file2)
        
        if not nodes1 or not nodes2:
            return 0.0

        vectorizer = CountVectorizer()
        try:
            vectorizer.fit([' '.join(nodes1 + nodes2)])
            vec1 = vectorizer.transform([' '.join(nodes1)])
            vec2 = vectorizer.transform([' '.join(nodes2)])
            similarity = cosine_similarity(vec1, vec2)[0][0]
        except Exception as e:
            print(f"Error en el cálculo de similitud: {e}")
            return 0.0

        if metrics_collector:
            metrics_collector.ast_similarities.append(similarity)
            metrics_collector.stop_timing()
            
        return similarity
    except Exception as e:
        print(f"Error al calcular similitud AST: {e}")
        return 0.0

def split_by_cases(dataset_dir):
    """
    Divide los casos por carpetas completas en lugar de archivos individuales
    """
    cases = list(dataset_dir.iterdir())
    train_cases, test_cases = train_test_split(
        cases,
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
    
    return np.array(features)

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

def get_label(original_file, file):
    """
    Calcula el porcentaje de plagio entre dos archivos
    """
    try:
        token_overlap = calculate_token_overlap(original_file, file)
        ast_similarity = calculate_ast_similarity(original_file, file)
        plagiarism_percentage = (token_overlap + ast_similarity) / 2.0 * 100
        return plagiarism_percentage
    except Exception as e:
        print(f"Error al calcular etiqueta: {e}")
        return 0.0

def save_metrics(metrics, model_dir):
    """
    Guarda métricas más detalladas incluyendo desviación estándar
    """
    try:
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
                "total_samples": int(metrics["total_samples"]),
                "training_samples": int(metrics["training_samples"]),
                "test_samples": int(metrics["test_samples"]),
                "timestamp": datetime.now().isoformat()
            }
        }

        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        metrics_path = model_dir / "metrics.json"
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
        print("\nMétricas guardadas en:", metrics_path)
        print("\nContenido del archivo de métricas:")
        with open(metrics_path, 'r', encoding='utf-8') as f:
            print(json.dumps(json.load(f), indent=2, ensure_ascii=False))
            
        return True
    except Exception as e:
        print(f"Error al guardar métricas: {e}")
        return False

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

def main():
    try:
        print("Iniciando entrenamiento del modelo mejorado...")
        print(f"Directorio actual: {os.getcwd()}")
        
        # Configurar directorios
        dataset_dir = get_dataset_path()
        model_dir = Path('model')
        model_dir.mkdir(exist_ok=True)
        
        print(f"\nUsando dataset en: {dataset_dir}")
        print(f"Usando directorio de modelo: {model_dir}")
        
        # Dividir casos por carpetas
        train_cases, test_cases = split_by_cases(dataset_dir)
        print(f"\nCasos de entrenamiento: {len(train_cases)}")
        print(f"Casos de prueba: {len(test_cases)}")
        
        features = []
        labels = []
        metrics_collector = MetricsCollector()
        processed_files = set()
        
        # Procesar casos de entrenamiento
        print("\nProcesando casos de entrenamiento...")
        for case in train_cases:
            try:
                original_file = load_original_file(case)
                plagiarized_files = load_plagiarized_files(case)
                non_plagiarized_files = load_non_plagiarized_files(case)
                
                for file in plagiarized_files + non_plagiarized_files:
                    if str(file) not in processed_files:
                        processed_files.add(str(file))
                        feature_vector = extract_features(original_file, file, metrics_collector)
                        features.append(feature_vector)
                        labels.append(get_label(original_file, file))
                        
            except Exception as e:
                print(f"Error procesando caso {case}: {e}")
                continue
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"\nCaracterísticas extraídas: {X.shape}")
        print(f"Etiquetas generadas: {len(y)}")
        
        # Entrenar modelo con validación cruzada
        print("\nEntrenando modelo con validación cruzada...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        cv_scores = train_with_cv(X, y, model)
        
        # Entrenar modelo final
        print("\nEntrenando modelo final...")
        model.fit(X, y)
        
        # Evaluar en conjunto de prueba
        print("\nEvaluando en conjunto de prueba...")
        test_features = []
        test_labels = []
        
        for case in test_cases:
            try:
                original_file = load_original_file(case)
                for file in load_plagiarized_files(case) + load_non_plagiarized_files(case):
                    if str(file) not in processed_files:
                        processed_files.add(str(file))
                        feature_vector = extract_features(original_file, file, metrics_collector)
                        test_features.append(feature_vector)
                        test_labels.append(get_label(original_file, file))
            except Exception as e:
                print(f"Error procesando caso de prueba {case}: {e}")
                continue
        
        X_test = np.array(test_features)
        y_test = np.array(test_labels)
        
        print(f"\nCaracterísticas de prueba: {X_test.shape}")
        print(f"Etiquetas de prueba: {len(y_test)}")
        
        y_pred = model.predict(X_test)
        
        # Calcular métricas completas
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "thresholds": evaluate_model(y_test, y_pred),
            "test_accuracy": cv_scores['test_accuracy'],
            "test_precision": cv_scores['test_precision'],
            "test_recall": cv_scores['test_recall'],
            "test_f1": cv_scores['test_f1'],
            "total_samples": len(y) + len(y_test),
            "training_samples": len(y),
            "test_samples": len(y_test)
        }
        
        # Guardar modelo y métricas
        print("\nGuardando modelo y métricas...")
        model_path = model_dir / 'classifier.joblib'
        dump(model, model_path)
        print(f"Modelo guardado en: {model_path}")
        
        save_metrics(metrics, model_dir)
        
        # Imprimir resultados
        print("\nMétricas del modelo:")
        print(f"MAE: {metrics['mae']:.4f}")
        print("\nResultados por umbral:")
        for threshold, scores in metrics['thresholds'].items():
            print(f"\nUmbral {threshold}:")
            for metric, value in scores.items():
                print(f"{metric}: {value:.4f}")
                
        print("\nResultados de validación cruzada:")
        print(f"Accuracy: {np.mean(metrics['test_accuracy']):.4f} (±{np.std(metrics['test_accuracy']):.4f})")
        print(f"Precision: {np.mean(metrics['test_precision']):.4f} (±{np.std(metrics['test_precision']):.4f})")
        print(f"Recall: {np.mean(metrics['test_recall']):.4f} (±{np.std(metrics['test_recall']):.4f})")
        print(f"F1-score: {np.mean(metrics['test_f1']):.4f} (±{np.std(metrics['test_f1']):.4f})")
        
        print("\nEntrenamiento completado exitosamente!")
        return 0
        
    except Exception as e:
        print(f"\nError general: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
