import os
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import javalang
from joblib import dump
import time
from collections import Counter
import math


class MetricsCollector:
    def __init__(self):
        self.case_metrics = {}
        self.execution_times = []
        self.token_similarities = []
        self.ast_similarities = []
        self.feature_similarity = []
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


def parse_ast_features(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    # Parse the AST
    tree = javalang.parse.parse(code)
    return tree


def parse_ast(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()
    tree = javalang.parse.parse(code)
    return [node.__class__.__name__ for _, node in tree]


def calculate_ast_similarity(file1, file2, metrics_collector=None):
    """
    Calcula la similitud AST entre dos archivos
    """
    if metrics_collector:
        metrics_collector.start_timing()

    try:
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


def calculate_features_similarity(original_file, file, metrics_collector=None):
    """
    New features similarity
    """
    if metrics_collector:
        metrics_collector.start_timing()

    try:
        features_original_file = extract_file_features(original_file)
        features_file = extract_file_features(file)

        feature_similarity_var = np.mean([
            np.abs(a - b) for a, b in zip(features_original_file, features_file)
            ])

        similarity = 1 - feature_similarity_var

        if metrics_collector:
            metrics_collector.feature_similarity.append(similarity)
            metrics_collector.stop_timing()

        return similarity

    except Exception as e:
        print(f"Error al calcular similitud de las nuevas features: {e}")
        return 0.0


def calculate_tree_depth(node):
    if not hasattr(node, 'children') or not node.children:
        return 1  # Leaf node
    return 1 + max(calculate_tree_depth(child) for child in node.children)


def calculate_number_of_nodes(tree):
    return len([node for _, node in tree])


def calculate_number_of_leaves(tree):
    return len([node for _, node in tree if isinstance(node, javalang.tree.Literal)])


def calculate_branching_factor(tree):
    nodes = [node for _, node in tree]
    children_count = [len(node.children) for node in nodes if hasattr(node, 'children')]
    return np.mean(children_count) if children_count else 0


def node_type_counts(tree):
    node_types = [node.__class__.__name__ for _, node in tree]
    return Counter(node_types)


def node_label_entropy(tree):
    node_types = [node.__class__.__name__ for _, node in tree]
    counts = Counter(node_types)
    total = sum(counts.values())
    probabilities = [count / total for count in counts.values()]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def calculate_most_common_paths(tree, depth=2):
    paths = []

    def traverse(node, path=[]):
        if len(path) == depth:
            paths.append(path)
            return
        if hasattr(node, 'children'):
            for child in node.children:
                traverse(child, path + [node.__class__.__name__])
    for _, node in tree:
        traverse(node)

    path_counts = Counter(tuple(path) for path in paths)
    return path_counts.most_common(5)  # Devuelve las 5 rutas más comunes


def count_variables(tree):
    return sum(1 for _, node in tree if isinstance(node, javalang.tree.VariableDeclarator))


def count_operations(tree):
    return sum(1 for _, node in tree if isinstance(node, javalang.tree.BinaryOperation))


def count_function_calls(tree):
    return sum(1 for _, node in tree if isinstance(node, javalang.tree.MethodInvocation))


def calculate_cyclomatic_complexity(tree):
    return sum(1 for _, node in tree if isinstance(node, javalang.tree.IfStatement)) + 1


def calculate_nesting_depth(tree):
    max_depth = 0

    def traverse(node, depth=0):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        if hasattr(node, 'children'):
            for child in node.children:
                traverse(child, depth + 1)

    for _, node in tree:
        traverse(node)
    return max_depth


def extract_file_features(file_path):
    tree = parse_ast_features(file_path)

    structural_features = [
        calculate_tree_depth(tree),
        calculate_number_of_nodes(tree),
        calculate_number_of_leaves(tree),
        calculate_branching_factor(tree)
    ]
    node_features = [
        node_label_entropy(tree),
        sum(node_type_counts(tree).values())
    ]
    semantic_features = [
        count_variables(tree),
        count_operations(tree),
        count_function_calls(tree)
    ]
    complexity_features = [
        calculate_cyclomatic_complexity(tree),
        calculate_nesting_depth(tree)
    ]

    features = structural_features + node_features + semantic_features + complexity_features

    assert all(isinstance(f, (int, float)) for f in features), "All features must be numeric"
    return features


def split_by_cases(dataset_dir):
    cases = list(dataset_dir.iterdir())

    # Separar casos con y sin plagio
    cases_with_plagiarism = []
    cases_without_plagiarism = []

    for case in cases:
        if (case / "plagiarized").exists():
            cases_with_plagiarism.append(case)
        else:
            cases_without_plagiarism.append(case)

    # Asegurar balance en train y test
    train_with_plag = cases_with_plagiarism[:int(0.8*len(cases_with_plagiarism))]
    test_with_plag = cases_with_plagiarism[int(0.8*len(cases_with_plagiarism)):]

    train_without_plag = cases_without_plagiarism[:int(0.8*len(cases_without_plagiarism))]
    test_without_plag = cases_without_plagiarism[int(0.8*len(cases_without_plagiarism)):]

    train_cases = train_with_plag + train_without_plag
    test_cases = test_with_plag + test_without_plag

    np.random.shuffle(train_cases)
    np.random.shuffle(test_cases)

    return train_cases, test_cases


def extract_features(original_file, comparison_file, metrics_collector=None):
    """
    Extrae características más completas para cada comparación
    """
    features = []

    # Token overlap
    token_overlap = calculate_token_overlap(original_file,
                                            comparison_file,
                                            metrics_collector)
    features.append(token_overlap)

    # AST similarity
    ast_similarity = calculate_ast_similarity(original_file,
                                              comparison_file,
                                              metrics_collector)

    features.append(ast_similarity)

    # Structural, Semantic, Node, Complexity similarity
    struct_semantic_features = calculate_features_similarity(original_file,
                                                             comparison_file,
                                                             metrics_collector)
    features.append(struct_semantic_features)

    return np.array(features)


def evaluate_model(y_true, y_pred):
    metrics = {}
    for threshold in [20, 30, 40, 50, 60]:
        y_true_binary = y_true > threshold
        y_pred_binary = y_pred > threshold

        # Verificar que hay ambas clases
        if len(np.unique(y_true_binary)) < 2:
            print(f"Advertencia: Solo una clase presente para umbral {threshold}")
            continue

        metrics[f'threshold_{threshold}'] = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary),
            'recall': recall_score(y_true_binary, y_pred_binary),
            'f1': f1_score(y_true_binary, y_pred_binary)
        }
    return metrics


def get_label(original_file, file):
    """
    Calcula el porcentaje de plagio entre dos archivos
    """
    try:
        token_overlap = calculate_token_overlap(original_file, file)
        ast_similarity = calculate_ast_similarity(original_file, file)
        feature_similarity = calculate_features_similarity(original_file, file)

        # Añadir un umbral más bajo para considerar plagio
        total_similarity = (token_overlap+ast_similarity+feature_similarity)
        plagiarism_percentage = total_similarity / 3.0 * 100

        # Verificar si el archivo está en carpeta de plagio o no
        is_plagiarized = "plagiarized" in str(file)

        # Ajustar el porcentaje según la ubicación del archivo
        if is_plagiarized and plagiarism_percentage < 50:
            plagiarism_percentage = min(plagiarism_percentage + 30, 100)
        elif not is_plagiarized and plagiarism_percentage > 20:
            plagiarism_percentage = max(plagiarism_percentage - 30, 0)

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
        if len(y) < 5:
            raise ValueError("Insuficientes datos para validación cruzada")

        cv_results = {
            'test_accuracy': [],
            'test_precision': [],
            'test_recall': [],
            'test_f1': []
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_val_fold)

            # Usar umbral de 50% para métricas
            y_val_binary = y_val_fold > 50
            y_pred_binary = y_pred_fold > 50

            cv_results['test_accuracy'].append(accuracy_score(y_val_binary, y_pred_binary))
            cv_results['test_precision'].append(precision_score(y_val_binary, y_pred_binary))
            cv_results['test_recall'].append(recall_score(y_val_binary, y_pred_binary))
            cv_results['test_f1'].append(f1_score(y_val_binary, y_pred_binary))

        return {k: np.array(v) for k, v in cv_results.items()}

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

        # Separar casos con y sin plagio
        cases_with_plagiarism = []
        cases_without_plagiarism = []

        for case in cases:
            plag_path = case / "plagiarized"
            non_plag_path = case / "non-plagiarized"

            if plag_path.exists() and list(plag_path.glob('**/*.java')):
                cases_with_plagiarism.append(case)
            if non_plag_path.exists() and list(non_plag_path.glob('**/*.java')):
                cases_without_plagiarism.append(case)

        print("\nDistribución de casos:")
        print(f"Casos con archivos plagiados: {len(cases_with_plagiarism)}")
        print(f"Casos con archivos no plagiados: {len(cases_without_plagiarism)}")

        # Dividir casos manteniendo balance
        train_with_plag = cases_with_plagiarism[:int(0.8*len(cases_with_plagiarism))]
        test_with_plag = cases_with_plagiarism[int(0.8*len(cases_with_plagiarism)):]

        train_without_plag = cases_without_plagiarism[:int(0.8*len(cases_without_plagiarism))]
        test_without_plag = cases_without_plagiarism[int(0.8*len(cases_without_plagiarism)):]

        train_cases = train_with_plag + train_without_plag
        test_cases = test_with_plag + test_without_plag

        np.random.shuffle(train_cases)
        np.random.shuffle(test_cases)

        print("\nDivisión de casos:")
        print(f"Entrenamiento - Con plagio: {len(train_with_plag)}, Sin plagio: {len(train_without_plag)}")
        print(f"Prueba - Con plagio: {len(test_with_plag)}, Sin plagio: {len(test_without_plag)}")

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

                # Procesar todos los archivos
                for file in (load_plagiarized_files(case) + load_non_plagiarized_files(case)):
                    if str(file) not in processed_files:
                        processed_files.add(str(file))
                        try:
                            feature_vector = extract_features(original_file, file, metrics_collector)
                            similarity = np.mean(feature_vector) * 100

                            if "non-plagiarized" in str(file):
                                # Aumentar reducción para no plagiados
                                plagiarism_score = max(similarity - 50, 0)
                            else:
                                # Reducir aumento para plagiados
                                plagiarism_score = min(similarity + 15, 100)
                            features.append(feature_vector)
                            labels.append(plagiarism_score)

                        except Exception as e:
                            print(f"Error procesando archivo {file}: {e}")
                            continue

            except Exception as e:
                print(f"Error procesando caso {case}: {e}")
                continue

        # Verificar que tenemos suficientes datos
        if len(features) < 10:
            raise ValueError("Insuficientes datos para entrenar el modelo")

        X = np.array(features)
        y = np.array(labels)

        # Verificar balance de clases con múltiples umbrales
        print("\nDistribución de clases por umbral:")
        for threshold in [30, 40, 50, 60, 70]:
            labels_binary = y > threshold
            positive_samples = np.sum(labels_binary)
            negative_samples = len(labels_binary) - positive_samples
            ratio = positive_samples / max(negative_samples, 1)
            print(f"Umbral {threshold}%:")
            print(f"  Positivos (>{threshold}%): {positive_samples}")
            print(f"  Negativos (≤{threshold}%): {negative_samples}")
            print(f"  Ratio: {ratio:.2f}")

        # Entrenar modelo con validación cruzada
        print("\nEntrenando modelo con validación cruzada...")
        model = RandomForestRegressor(
            n_estimators=200,  # Aumentar número de árboles
            max_depth=10,      # Limitar profundidad
            random_state=42,
            n_jobs=-1
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

                # Procesar todos los archivos de prueba
                for file in (load_plagiarized_files(case) + load_non_plagiarized_files(case)):
                    if str(file) not in processed_test_files:
                        processed_test_files.add(str(file))
                        try:
                            feature_vector = extract_features(original_file, file, metrics_collector)
                            similarity = np.mean(feature_vector) * 100

                            if "non-plagiarized" in str(file):
                                plagiarism_score = max(similarity - 40, 0)  # Reducir más para no plagiados
                            else:
                                plagiarism_score = min(similarity + 20, 100)  # Aumentar menos para plagiados

                            test_features.append(feature_vector)
                            test_labels.append(plagiarism_score)

                        except Exception as e:
                            print(f"Error procesando archivo de prueba {file}: {e}")
                            continue

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
