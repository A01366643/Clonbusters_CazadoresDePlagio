from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import javalang
from joblib import dump


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


def calculate_token_overlap(file1, file2):
    tokens1 = set(tokenize(file1))
    tokens2 = set(tokenize(file2))
    overlap = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    return overlap


def calculate_ast_similarity(file1, file2):
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

    return cosine_similarity(vec1, vec2)[0][0]


def get_label(original_file, file):
    token_overlap = calculate_token_overlap(original_file, file)
    ast_similarity = calculate_ast_similarity(original_file, file)
    
    # Calcular el porcentaje de plagio basado solo en tokens y AST
    plagiarism_percentage = (token_overlap + ast_similarity) / 2.0 * 100
    return plagiarism_percentage

def main():
    print("Iniciando entrenamiento del modelo Clonbusters...")
    
    # Definir rutas correctamente
    current_dir = Path(__file__).parent
    model_dir = current_dir  # Directorio donde se guardarán los modelos
    dataset_dir = current_dir.parent / "data" / "IR-Plag-Dataset"
    
    # Crear el directorio model si no existe
    model_dir.mkdir(exist_ok=True)
    
    print(f"Directorio actual: {current_dir}")
    print(f"Directorio del modelo: {model_dir}")
    print(f"Directorio del dataset: {dataset_dir}")

    # Verificar que el dataset existe
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio del dataset: {dataset_dir}")
    
    print(f"Cargando dataset desde: {dataset_dir}")
    cases = load_dataset(dataset_dir)
    print(f"Casos encontrados: {len(cases)}")

    features = []
    labels = []
    text_samples = []  # Para entrenar el vectorizador

    for case in cases:
        print(f"Procesando caso: {case}")
        original_file = load_original_file(case)
        plagiarized_files = load_plagiarized_files(case)
        non_plagiarized_files = load_non_plagiarized_files(case)
        
        # Recolectar muestras de texto para el vectorizador
        with open(original_file, 'r') as f:
            text_samples.append(f.read())
        
        for file in plagiarized_files + non_plagiarized_files:
            print(f"Analizando archivo: {file}")
            token_overlap = calculate_token_overlap(original_file, file)
            ast_similarity = calculate_ast_similarity(original_file, file)
            
            with open(file, 'r') as f:
                text_samples.append(f.read())

            # Solo incluir token_overlap y ast_similarity en las características
            features.append([token_overlap, ast_similarity])
            labels.append(get_label(original_file, file))

    # Entrenar y guardar el vectorizador
    print("Entrenando vectorizador...")
    vectorizer = CountVectorizer()
    vectorizer.fit(text_samples)
    
    # Guardar el vectorizador con manejo de errores
    try:
        vectorizer_path = model_dir / "vectorizer.joblib"
        print(f"Guardando vectorizador en: {vectorizer_path}")
        dump(vectorizer, vectorizer_path)
        print("¡Vectorizador guardado exitosamente!")
    except Exception as e:
        print(f"Error al guardar el vectorizador: {e}")
        raise

    # Entrenar y guardar el clasificador
    X = np.array(features)
    y = np.array(labels)

    print("Dividiendo dataset en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Entrenando modelo RandomForestRegressor...")
    classifier = RandomForestRegressor(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    print("Evaluando modelo...")
    y_pred = classifier.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Error Absoluto Medio: {mae}')

    # Guardar el clasificador con manejo de errores
    try:
        classifier_path = model_dir / "classifier.joblib"
        print(f"Guardando clasificador en: {classifier_path}")
        dump(classifier, classifier_path)
        print("¡Clasificador guardado exitosamente!")
    except Exception as e:
        print(f"Error al guardar el clasificador: {e}")
        raise

    # Verificar que los archivos se crearon correctamente
    print("\nVerificando archivos generados:")
    vectorizer_path = model_dir / "vectorizer.joblib"
    classifier_path = model_dir / "classifier.joblib"
    for file_path in [vectorizer_path, classifier_path]:
        if file_path.exists():
            print(f"✅ {file_path.name} creado correctamente")
        else:
            print(f"❌ {file_path.name} no se creó")
            
    print("\nResumen de archivos generados:")
    print(f"Vectorizador: {vectorizer_path}")
    print(f"Clasificador: {classifier_path}")

if __name__ == "__main__":
    main()
