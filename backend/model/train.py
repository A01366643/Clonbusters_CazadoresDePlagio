from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import javalang
from joblib import dump
import time
from datetime import datetime
import pandas as pd

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

def load_dataset(path):
    full_paths = list(path.iterdir())
    return full_paths

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
    # Convert continuous predictions to binary for classification metrics
    threshold = 50  # Consider it plagiarism if similarity is > 50%
    y_pred_binary = (y_pred > threshold).astype(int)
    y_test_binary = (y_test > threshold).astype(int)
    
    report = {
        "Model Performance": {
            "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
            "Accuracy": accuracy_score(y_test_binary, y_pred_binary),
            "Precision": precision_score(y_test_binary, y_pred_binary),
            "Recall": recall_score(y_test_binary, y_pred_binary),
            "F1 Score": f1_score(y_test_binary, y_pred_binary),
            "Confusion Matrix": confusion_matrix(y_test_binary, y_pred_binary).tolist()
        },
        "Analysis Methods": {
            "Token Analysis": {
                "Mean Similarity": np.mean(metrics_collector.token_similarities),
                "Std Similarity": np.std(metrics_collector.token_similarities),
                "Min Similarity": np.min(metrics_collector.token_similarities),
                "Max Similarity": np.max(metrics_collector.token_similarities)
            },
            "AST Analysis": {
                "Mean Similarity": np.mean(metrics_collector.ast_similarities),
                "Std Similarity": np.std(metrics_collector.ast_similarities),
                "Min Similarity": np.min(metrics_collector.ast_similarities),
                "Max Similarity": np.max(metrics_collector.ast_similarities)
            }
        },
        "Execution Time": {
            "Mean Processing Time": np.mean(metrics_collector.execution_times),
            "Total Processing Time": np.sum(metrics_collector.execution_times),
            "Files Processed": len(metrics_collector.execution_times)
        },
        "Case Analysis": metrics_collector.case_metrics
    }
    
    # Save detailed report
    report_path = Path(__file__).parent / "metrics_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Plagiarism Detection Metrics Report\n")
        f.write(f"Generated on: {datetime.now()}\n\n")
        
        # Model Performance
        f.write("1. Model Performance Metrics\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean Absolute Error: {report['Model Performance']['Mean Absolute Error']:.4f}\n")
        f.write(f"Accuracy: {report['Model Performance']['Accuracy']:.4f}\n")
        f.write(f"Precision: {report['Model Performance']['Precision']:.4f}\n")
        f.write(f"Recall: {report['Model Performance']['Recall']:.4f}\n")
        f.write(f"F1 Score: {report['Model Performance']['F1 Score']:.4f}\n\n")
        
        # Confusion Matrix
        f.write("Confusion Matrix:\n")
        cm = report['Model Performance']['Confusion Matrix']
        f.write(f"[[{cm[0][0]}, {cm[0][1]}]\n")
        f.write(f" [{cm[1][0]}, {cm[1][1]}]]\n\n")
        
        # Analysis Methods Comparison
        f.write("2. Analysis Methods Comparison\n")
        f.write("-" * 30 + "\n")
        f.write("Token Analysis:\n")
        for key, value in report['Analysis Methods']['Token Analysis'].items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\nAST Analysis:\n")
        for key, value in report['Analysis Methods']['AST Analysis'].items():
            f.write(f"{key}: {value:.4f}\n")
        
        # Execution Times
        f.write("\n3. Execution Time Analysis\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean Processing Time: {report['Execution Time']['Mean Processing Time']:.4f} seconds\n")
        f.write(f"Total Processing Time: {report['Execution Time']['Total Processing Time']:.4f} seconds\n")
        f.write(f"Files Processed: {report['Execution Time']['Files Processed']}\n")
        
        # Case Analysis
        f.write("\n4. Case-by-Case Analysis\n")
        f.write("-" * 30 + "\n")
        for case, metrics in report['Case Analysis'].items():
            f.write(f"\nCase {case}:\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value}\n")
    
    return report

def main():
    print("Iniciando entrenamiento del modelo Clonbusters con métricas...")
    
    current_dir = Path(__file__).parent
    model_dir = current_dir
    dataset_dir = current_dir.parent / "data" / "IR-Plag-Dataset"
    model_dir.mkdir(exist_ok=True)
    
    metrics_collector = MetricsCollector()
    
    print(f"Cargando dataset desde: {dataset_dir}")
    cases = load_dataset(dataset_dir)
    
    features = []
    labels = []
    text_samples = []
    file_paths = []  # Para tracking

    for case_idx, case in enumerate(cases, 1):
        print(f"Procesando caso {case_idx}/{len(cases)}: {case}")
        
        case_metrics = {
            "total_files": 0,
            "plagiarized_files": 0,
            "non_plagiarized_files": 0,
            "avg_similarity": 0,
            "processing_time": 0
        }
        
        metrics_collector.start_timing()
        
        original_file = load_original_file(case)
        plagiarized_files = load_plagiarized_files(case)
        non_plagiarized_files = load_non_plagiarized_files(case)
        
        case_metrics["plagiarized_files"] = len(plagiarized_files)
        case_metrics["non_plagiarized_files"] = len(non_plagiarized_files)
        case_metrics["total_files"] = len(plagiarized_files) + len(non_plagiarized_files)
        
        with open(original_file, 'r') as f:
            text_samples.append(f.read())
        
        similarities = []
        for file in plagiarized_files + non_plagiarized_files:
            token_overlap = calculate_token_overlap(original_file, file, metrics_collector)
            ast_similarity = calculate_ast_similarity(original_file, file, metrics_collector)
            
            with open(file, 'r') as f:
                text_samples.append(f.read())
            
            features.append([token_overlap, ast_similarity])
            label = get_label(original_file, file)
            labels.append(label)
            similarities.append(label)
            file_paths.append(file)
        
        case_metrics["avg_similarity"] = np.mean(similarities)
        case_metrics["processing_time"] = metrics_collector.stop_timing()
        metrics_collector.case_metrics[f"case_{case_idx}"] = case_metrics
    
    # Entrenamiento y evaluación
    X = np.array(features)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Entrenando modelo...")
    classifier = RandomForestRegressor(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    print("Evaluando modelo...")
    y_pred = classifier.predict(X_test)
    
    # Generar reporte de métricas
    print("Generando reporte de métricas...")
    metrics_report = generate_metrics_report(metrics_collector, y_test, y_pred, file_paths, classifier)
    
    # Guardar modelo y vectorizador
    try:
        classifier_path = model_dir / "classifier.joblib"
        dump(classifier, classifier_path)
        print(f"Modelo guardado en: {classifier_path}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
    
    print("\nResumen de métricas principales:")
    print(f"MAE: {metrics_report['Model Performance']['Mean Absolute Error']:.4f}")
    print(f"Accuracy: {metrics_report['Model Performance']['Accuracy']:.4f}")
    print(f"F1 Score: {metrics_report['Model Performance']['F1 Score']:.4f}")
    print(f"Tiempo promedio de procesamiento: {metrics_report['Execution Time']['Mean Processing Time']:.4f} segundos")
    print("\nReporte completo guardado en: metrics_report.txt")

if __name__ == "__main__":
    main()
