import os
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
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

def main():
    try:
        print("Iniciando entrenamiento del modelo Clonbusters...")
        print(f"Directorio actual: {os.getcwd()}")
        print("Contenido del directorio actual:")
        os.system('ls -la')
        
        # Configurar directorios
        try:
            dataset_dir = get_dataset_path()
            model_dir = Path('model')
            model_dir.mkdir(exist_ok=True)
            
            print(f"Usando dataset en: {dataset_dir}")
            print(f"Usando directorio de modelo: {model_dir}")
        except Exception as e:
            print(f"Error en la configuración de directorios: {e}")
            return 1
            
        metrics_collector = MetricsCollector()
        
        # Cargar y procesar dataset
        print("\nCargando casos del dataset...")
        try:
            cases = load_dataset(dataset_dir)
            print(f"Casos encontrados: {len(cases)}")
        except Exception as e:
            print(f"Error al cargar dataset: {e}")
            return 1
        
        features = []
        labels = []
        file_paths = []
        
        # Procesar casos
        for case_idx, case in enumerate(cases, 1):
            try:
                print(f"\nProcesando caso {case_idx}/{len(cases)}: {case}")
                
                original_file = load_original_file(case)
                plagiarized_files = load_plagiarized_files(case)
                non_plagiarized_files = load_non_plagiarized_files(case)
                
                for file in plagiarized_files + non_plagiarized_files:
                    try:
                        token_overlap = calculate_token_overlap(original_file, file, metrics_collector)
                        ast_similarity = calculate_ast_similarity(original_file, file, metrics_collector)
                        
                        features.append([token_overlap, ast_similarity])
                        labels.append(get_label(original_file, file))
                        file_paths.append(str(file))
                    except Exception as e:
                        print(f"Error procesando archivo {file}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error en caso {case}: {e}")
                continue
        
        if not features or not labels:
            print("No se pudieron extraer características. Abortando.")
            return 1
        
        # Entrenamiento y evaluación
        X = np.array(features)
        y = np.array(labels)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("\nEntrenando modelo...")
        classifier = RandomForestRegressor(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        
        print("Evaluando modelo...")
        y_pred = classifier.predict(X_test)
        
        # Calcular y guardar métricas
        metrics = {
            "model_performance": {
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "accuracy": float(accuracy_score(y_test > 50, y_pred > 50)),
                "precision": float(precision_score(y_test > 50, y_pred > 50)),
                "recall": float(recall_score(y_test > 50, y_pred > 50)),
                "f1_score": float(f1_score(y_test > 50, y_pred > 50))
            },
            "training_info": {
                "total_samples": len(y),
                "training_samples": len(y_train),
                "test_samples": len(y_test),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Guardar modelo y métricas
        try:
            model_path = model_dir / "classifier.joblib"
            metrics_path = model_dir / "metrics.json"
            
            print(f"\nGuardando modelo en {model_path}")
            dump(classifier, model_path)
            
            print(f"Guardando métricas en {metrics_path}")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print("\nMétricas del modelo:")
            print(f"MAE: {metrics['model_performance']['mae']:.4f}")
            print(f"Accuracy: {metrics['model_performance']['accuracy']:.4f}")
            print(f"F1 Score: {metrics['model_performance']['f1_score']:.4f}")
            
        except Exception as e:
            print(f"Error al guardar modelo o métricas: {e}")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"Error general: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)