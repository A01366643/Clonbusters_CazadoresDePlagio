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
    cases = list(dataset_dir.iterdir())
    if len(cases) < 2:
        raise ValueError("Insuficientes casos para dividir")
        
    # Asegurar aleatoriedad en la división
    np.random.shuffle(cases)
    
    # Dividir manteniendo la proporción de casos plagiados/no plagiados
    train_size = int(0.8 * len(cases))
    train_cases = cases[:train_size]
    test_cases = cases[train_size:]
    
    print(f"División de casos - Train: {len(train_cases)}, Test: {len(test_cases)}")
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
    # Verificar que tenemos métricas válidas
    if metrics["test_accuracy"] is None or np.isnan(metrics["test_accuracy"]).any():
        metrics["cross_validation"] = {
            "accuracy": {"mean": 0, "std": 0},
            "precision": {"mean": 0, "std": 0},
            "recall": {"mean": 0, "std": 0},
            "f1_score": {"mean": 0, "std": 0}
        }

def train_with_cv(X, y, model):
    try:
        # Asegurarnos de que tenemos suficientes datos para cada fold
        if len(y) < 5:
            raise ValueError("Insuficientes datos para validación cruzada")
            
        # Convertir las etiquetas continuas a binarias para clasificación
        y_binary = y > 50  # o el umbral que decidamos
        
        cv_scores = cross_validate(
            model, 
            X, 
            y_binary,
            cv=5,
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            error_score='raise'  # Para ver errores explícitos
        )
        return cv_scores
    except Exception as e:
        print(f"Error en validación cruzada: {e}")
        return None

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
        
        # Cargar y verificar dataset
        try:
            cases = load_dataset(dataset_dir)
            if len(cases) < 2:
                raise ValueError("Insuficientes casos en el dataset")
            print(f"\nTotal de casos encontrados: {len(cases)}")
        except Exception as e:
            print(f"Error al cargar dataset: {e}")
            return 1
        
        # Dividir casos por carpetas asegurando distribución balanceada
        try:
            np.random.shuffle(cases)
            train_size = int(0.8 * len(cases))
            train_cases = cases[:train_size]
            test_cases = cases[train_size:]
            
            print(f"\nCasos de entrenamiento: {len(train_cases)}")
            print(f"Casos de prueba: {len(test_cases)}")
        except Exception as e:
            print(f"Error en la división de casos: {e}")
            return 1
        
        features = []
        labels = []
        metrics_collector = MetricsCollector()
        processed_files = set()
        
        # Procesar casos de entrenamiento
        print("\nProcesando casos de entrenamiento...")
        for idx, case in enumerate(train_cases, 1):
            try:
                print(f"Procesando caso {idx}/{len(train_cases)}: {case.name}")
                
                original_file = load_original_file(case)
                plagiarized_files = load_plagiarized_files(case)
                non_plagiarized_files = load_non_plagiarized_files(case)
                
                # Procesar archivos plagiados
                for file in plagiarized_files:
                    if str(file) not in processed_files:
                        processed_files.add(str(file))
                        feature_vector = extract_features(original_file, file, metrics_collector)
                        features.append(feature_vector)
                        labels.append(get_label(original_file, file))
                        
                # Procesar archivos no plagiados
                for file in non_plagiarized_files:
                    if str(file) not in processed_files:
                        processed_files.add(str(file))
                        feature_vector = extract_features(original_file, file, metrics_collector)
                        features.append(feature_vector)
                        labels.append(get_label(original_file, file))
                        
            except Exception as e:
                print(f"Error procesando caso {case}: {e}")
                continue
        
        # Verificar que tenemos suficientes datos
        if len(features) < 10:
            raise ValueError("Insuficientes datos para entrenar el modelo")
            
        X = np.array(features)
        y = np.array(labels)
        
        # Verificar balance de clases
        labels_binary = y > 50
        positive_samples = np.sum(labels_binary)
        negative_samples = len(labels_binary) - positive_samples
        
        print(f"\nEstadísticas del conjunto de entrenamiento:")
        print(f"Características extraídas: {X.shape}")
        print(f"Casos positivos (plagio > 50%): {positive_samples}")
        print(f"Casos negativos (plagio ≤ 50%): {negative_samples}")
        print(f"Ratio positivos/negativos: {positive_samples/negative_samples:.2f}")
        
        # Entrenar modelo con validación cruzada
        print("\nEntrenando modelo con validación cruzada...")
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # Usar todos los cores disponibles
        )
        
        try:
            cv_scores = train_with_cv(X, y, model)
            if cv_scores is None:
                print("Advertencia: La validación cruzada falló, continuando con entrenamiento normal")
                cv_scores = {
                    'test_accuracy': np.array([0]),
                    'test_precision': np.array([0]),
                    'test_recall': np.array([0]),
                    'test_f1': np.array([0])
                }
        except Exception as e:
            print(f"Error en validación cruzada: {e}")
            cv_scores = {
                'test_accuracy': np.array([0]),
                'test_precision': np.array([0]),
                'test_recall': np.array([0]),
                'test_f1': np.array([0])
            }
        
        # Entrenar modelo final
        print("\nEntrenando modelo final...")
        model.fit(X, y)
        
        # Evaluar en conjunto de prueba
        print("\nEvaluando en conjunto de prueba...")
        test_features = []
        test_labels = []
        processed_test_files = set()
        
        for idx, case in enumerate(test_cases, 1):
            try:
                print(f"Procesando caso de prueba {idx}/{len(test_cases)}: {case.name}")
                original_file = load_original_file(case)
                
                for file in load_plagiarized_files(case) + load_non_plagiarized_files(case):
                    if str(file) not in processed_test_files:
                        processed_test_files.add(str(file))
                        feature_vector = extract_features(original_file, file, metrics_collector)
                        test_features.append(feature_vector)
                        test_labels.append(get_label(original_file, file))
                        
            except Exception as e:
                print(f"Error procesando caso de prueba {case}: {e}")
                continue
        
        if len(test_features) == 0:
            raise ValueError("No se pudieron extraer características del conjunto de prueba")
            
        X_test = np.array(test_features)
        y_test = np.array(test_labels)
        
        print(f"\nEstadísticas del conjunto de prueba:")
        print(f"Características de prueba: {X_test.shape}")
        print(f"Etiquetas de prueba: {len(y_test)}")
        
        # Realizar predicciones
        y_pred = model.predict(X_test)
        
        # Calcular métricas completas
        try:
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
        except Exception as e:
            print(f"Error calculando métricas: {e}")
            return 1
        
        # Guardar modelo y métricas
        print("\nGuardando modelo y métricas...")
        try:
            model_path = model_dir / 'classifier.joblib'
            dump(model, model_path)
            print(f"Modelo guardado en: {model_path}")
            
            if not save_metrics(metrics, model_dir):
                print("Advertencia: No se pudieron guardar las métricas completas")
        except Exception as e:
            print(f"Error guardando modelo o métricas: {e}")
            return 1
        
        # Imprimir resultados detallados
        print("\nMétricas del modelo:")
        print(f"MAE: {metrics['mae']:.4f}")
        
        print("\nResultados por umbral:")
        for threshold, scores in metrics['thresholds'].items():
            print(f"\nUmbral {threshold}:")
            for metric, value in scores.items():
                print(f"{metric}: {value:.4f}")
        
        if cv_scores is not None:
            print("\nResultados de validación cruzada:")
            print(f"Accuracy: {np.mean(cv_scores['test_accuracy']):.4f} (±{np.std(cv_scores['test_accuracy']):.4f})")
            print(f"Precision: {np.mean(cv_scores['test_precision']):.4f} (±{np.std(cv_scores['test_precision']):.4f})")
            print(f"Recall: {np.mean(cv_scores['test_recall']):.4f} (±{np.std(cv_scores['test_recall']):.4f})")
            print(f"F1-score: {np.mean(cv_scores['test_f1']):.4f} (±{np.std(cv_scores['test_f1']):.4f})")
        
        print("\nEntrenamiento completado exitosamente!")
        return 0
        
    except Exception as e:
        print(f"\nError general: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
