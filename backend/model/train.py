import os
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
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
    # Intentar obtener la ruta del dataset desde la variable de entorno
    dataset_path = os.getenv('DATASET_PATH')
    if dataset_path:
        return Path(dataset_path)
    
    # Si no está en la variable de entorno, intentar encontrarlo relativamente
    current_dir = Path(__file__).parent
    possible_paths = [
        current_dir.parent / "data" / "IR-Plag-Dataset",
        current_dir.parent.parent / "data" / "IR-Plag-Dataset",
        Path("data/IR-Plag-Dataset"),
        Path("../data/IR-Plag-Dataset"),
        Path("../../data/IR-Plag-Dataset")
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError("No se pudo encontrar el directorio del dataset")

def get_model_dir():
    # Intentar usar el directorio actual del script
    current_dir = Path(__file__).parent
    model_dir = current_dir
    
    # Si estamos en GitHub Actions, usar el directorio especificado
    if os.getenv('GITHUB_ACTIONS'):
        model_dir = Path('model')
    
    # Crear el directorio si no existe
    model_dir.mkdir(exist_ok=True)
    return model_dir

def load_dataset(path):
    try:
        full_paths = list(path.iterdir())
        print(f"Archivos encontrados en {path}:")
        for p in full_paths:
            print(f"  - {p}")
        return full_paths
    except Exception as e:
        print(f"Error al cargar el dataset desde {path}: {e}")
        raise

def load_original_file(path):
    original_file_path_directory = str(path) + "/original"
    original_file_path_file = Path(original_file_path_directory)
    original_file = list(original_file_path_file.iterdir())
    return original_file[0]

def load_plagiarized_files(path):
    plagiarized_path = str(path) + "/plagiarized"
    plagiarized_path = Path(plagiarized_path)
    plagiarized_L_directories = list(plagiarized_path.iterdir())
    plagiarized_files_paths = []
    for path in plagiarized_L_directories:
        plagiarized_files_paths = plagiarized_files_paths + list(path.iterdir())

    plagiarized_files = []
    for path in plagiarized_files_paths:
        plagiarized_files = plagiarized_files + list(path.iterdir())

    return plagiarized_files

def load_non_plagiarized_files(path):
    non_plagiarized_path = str(path) + "/non-plagiarized"
    non_plagiarized_path = Path(non_plagiarized_path)
    non_plagiarized_inner = list(non_plagiarized_path.iterdir())

    non_plagiarized_files = []
    for path in non_plagiarized_inner:
        non_plagiarized_files = non_plagiarized_files + list(path.iterdir())

    return non_plagiarized_files

def tokenize(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    tokens = list(javalang.tokenizer.tokenize(code))
    return [token.value for token in tokens]

def calculate_token_overlap(file1, file2, metrics_collector=None):
    if metrics_collector:
        metrics_collector.start_timing()
    
    tokens1 = set(tokenize(file1))
    tokens2 = set(tokenize(file2))
    overlap = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    
    if metrics_collector:
        execution_time = metrics_collector.stop_timing()
        metrics_collector.token_similarities.append(overlap)
    
    return overlap

def calculate_ast_similarity(file1, file2, metrics_collector=None):
    if metrics_collector:
        metrics_collector.start_timing()
        
    def parse_ast(file_path):
        with open(file_path, 'r') as file:
            code = file.read()
        tree = javalang.parse.parse(code)
        return tree

    tree1 = parse_ast(file1)
    tree2 = parse_ast(file2)

    nodes1 = [node.__class__.__name__ for _, node in tree1]
    nodes2 = [node.__class__.__name__ for _, node in tree2]
    vectorizer = CountVectorizer().fit(nodes1 + nodes2)

    vec1 = vectorizer.transform([' '.join(nodes1)]).toarray()
    vec2 = vectorizer.transform([' '.join(nodes2)]).toarray()
    
    similarity = cosine_similarity(vec1, vec2)[0][0]
    
    if metrics_collector:
        execution_time = metrics_collector.stop_timing()
        metrics_collector.ast_similarities.append(similarity)
    
    return similarity

def get_label(original_file, file):
    token_overlap = calculate_token_overlap(original_file, file)
    ast_similarity = calculate_ast_similarity(original_file, file)
    plagiarism_percentage = (token_overlap + ast_similarity) / 2.0 * 100
    return plagiarism_percentage

def generate_metrics_report(metrics_collector, y_test, y_pred, test_files, model):
    try:
        # Convert continuous predictions to binary for classification metrics
        threshold = 50  # Consider it plagiarism if similarity is > 50%
        y_pred_binary = (y_pred > threshold).astype(int)
        y_test_binary = (y_test > threshold).astype(int)
        
        report = {
            "Model Performance": {
                "Mean Absolute Error": float(mean_absolute_error(y_test, y_pred)),
                "Accuracy": float(accuracy_score(y_test_binary, y_pred_binary)),
                "Precision": float(precision_score(y_test_binary, y_pred_binary)),
                "Recall": float(recall_score(y_test_binary, y_pred_binary)),
                "F1 Score": float(f1_score(y_test_binary, y_pred_binary))
            },
            "Analysis Methods": {
                "Token Analysis": {
                    "Mean Similarity": float(np.mean(metrics_collector.token_similarities)) if metrics_collector.token_similarities else 0.0,
                    "Std Similarity": float(np.std(metrics_collector.token_similarities)) if metrics_collector.token_similarities else 0.0,
                    "Min Similarity": float(np.min(metrics_collector.token_similarities)) if metrics_collector.token_similarities else 0.0,
                    "Max Similarity": float(np.max(metrics_collector.token_similarities)) if metrics_collector.token_similarities else 0.0
                },
                "AST Analysis": {
                    "Mean Similarity": float(np.mean(metrics_collector.ast_similarities)) if metrics_collector.ast_similarities else 0.0,
                    "Std Similarity": float(np.std(metrics_collector.ast_similarities)) if metrics_collector.ast_similarities else 0.0,
                    "Min Similarity": float(np.min(metrics_collector.ast_similarities)) if metrics_collector.ast_similarities else 0.0,
                    "Max Similarity": float(np.max(metrics_collector.ast_similarities)) if metrics_collector.ast_similarities else 0.0
                }
            },
            "Execution Time": {
                "Mean Processing Time": float(np.mean(metrics_collector.execution_times)) if metrics_collector.execution_times else 0.0,
                "Total Processing Time": float(np.sum(metrics_collector.execution_times)) if metrics_collector.execution_times else 0.0,
                "Files Processed": len(metrics_collector.execution_times)
            },
            "Case Analysis": metrics_collector.case_metrics
        }
        
        # Guardar reporte en formato JSON para mejor compatibilidad
        import json
        report_path = Path(__file__).parent / "metrics_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        
        # También guardar una versión legible en texto
        txt_report_path = Path(__file__).parent / "metrics_report.txt"
        with open(txt_report_path, "w") as f:
            f.write(f"Plagiarism Detection Metrics Report\n")
            f.write(f"Generated on: {datetime.now()}\n\n")
            
            # Model Performance
            f.write("1. Model Performance Metrics\n")
            f.write("-" * 30 + "\n")
            for metric, value in report["Model Performance"].items():
                f.write(f"{metric}: {value:.4f}\n")
            
            # Analysis Methods
            f.write("\n2. Analysis Methods Comparison\n")
            f.write("-" * 30 + "\n")
            for method, metrics in report["Analysis Methods"].items():
                f.write(f"\n{method}:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
            
            # Execution Times
            f.write("\n3. Execution Time Analysis\n")
            f.write("-" * 30 + "\n")
            for metric, value in report["Execution Time"].items():
                if isinstance(value, float):
                    f.write(f"{metric}: {value:.4f} seconds\n")
                else:
                    f.write(f"{metric}: {value}\n")
            
            # Case Analysis
            f.write("\n4. Case-by-Case Analysis\n")
            f.write("-" * 30 + "\n")
            for case, metrics in report["Case Analysis"].items():
                f.write(f"\nCase {case}:\n")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        f.write(f"{metric_name}: {metric_value:.4f}\n")
                    else:
                        f.write(f"{metric_name}: {metric_value}\n")
        
        return report
    except Exception as e:
        print(f"Error al generar el reporte de métricas: {str(e)}")
        # Crear un reporte mínimo en caso de error
        error_report = {
            "error": str(e),
            "status": "failed",
            "timestamp": str(datetime.now())
        }
        
        # Guardar el reporte de error
        error_report_path = Path(__file__).parent / "error_report.json"
        with open(error_report_path, "w") as f:
            json.dump(error_report, f, indent=4)
        
        return error_report

def main():
    try:
        print("Iniciando entrenamiento del modelo Clonbusters...")
        print(f"Directorio actual: {os.getcwd()}")
        print(f"Contenido del directorio actual:")
        os.system('ls -la')
        
        # Obtener rutas
        try:
            dataset_dir = get_dataset_path()
            print(f"Usando dataset en: {dataset_dir}")
        except FileNotFoundError as e:
            print(f"Error al encontrar dataset: {e}")
            return 1
        
        try:
            model_dir = get_model_dir()
            print(f"Usando directorio de modelo: {model_dir}")
        except Exception as e:
            print(f"Error al configurar directorio del modelo: {e}")
            return 1
            
        metrics_collector = MetricsCollector()
        
        # Verificar existencia del dataset
        if not dataset_dir.exists():
            print(f"El directorio del dataset no existe: {dataset_dir}")
            print("Contenido del directorio padre:")
            parent_dir = dataset_dir.parent
            if parent_dir.exists():
                print(list(parent_dir.iterdir()))
            return 1
            
        print("Cargando casos del dataset...")
        cases = load_dataset(dataset_dir)
        print(f"Casos encontrados: {len(cases)}")
        
        features = []
        labels = []
        file_paths = []
        
        for case_idx, case in enumerate(cases, 1):
            try:
                print(f"\nProcesando caso {case_idx}/{len(cases)}: {case}")
                
                # Procesar archivos originales
                original_file = load_original_file(case)
                if not original_file.exists():
                    print(f"Archivo original no encontrado en {case}")
                    continue
                    
                # Procesar archivos plagiarized y non-plagiarized
                plagiarized_files = load_plagiarized_files(case)
                non_plagiarized_files = load_non_plagiarized_files(case)
                
                # Calcular características
                for file in plagiarized_files + non_plagiarized_files:
                    try:
                        token_overlap = calculate_token_overlap(original_file, file, metrics_collector)
                        ast_similarity = calculate_ast_similarity(original_file, file, metrics_collector)
                        
                        features.append([token_overlap, ast_similarity])
                        label = get_label(original_file, file)
                        labels.append(label)
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
            
        # Entrenamiento
        X = np.array(features)
        y = np.array(labels)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("\nEntrenando modelo...")
        classifier = RandomForestRegressor(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        
        print("Evaluando modelo...")
        y_pred = classifier.predict(X_test)
        
        # Guardar modelo
        try:
            model_path = model_dir / "classifier.joblib"
            print(f"Guardando modelo en {model_path}")
            dump(classifier, model_path)
            print("Modelo guardado exitosamente")
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
            return 1
        
        # Guardar métricas
        try:
            metrics = {
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "accuracy": float(accuracy_score(y_test > 50, y_pred > 50)),
                "f1_score": float(f1_score(y_test > 50, y_pred > 50)),
                "total_samples": len(y),
                "training_samples": len(y_train),
                "test_samples": len(y_test),
                "timestamp": datetime.now().isoformat()
            }
            
            metrics_path = model_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Métricas guardadas en {metrics_path}")
            
            # Imprimir métricas principales
            print("\nMétricas del modelo:")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            print(f"Error al guardar métricas: {e}")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"Error general: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
