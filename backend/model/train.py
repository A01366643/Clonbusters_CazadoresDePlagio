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


def calculate_semantic_similarity(file1, file2):
    with open(file1, 'r') as f:
        text1 = f.read()
    with open(file2, 'r') as f:
        text2 = f.read()

    vectorizer = CountVectorizer().fit([text1, text2])
    vec1 = vectorizer.transform([text1]).toarray()
    vec2 = vectorizer.transform([text2]).toarray()

    return cosine_similarity(vec1, vec2)[0][0]


def get_label(original_file, file):
    token_overlap = calculate_token_overlap(original_file, file)
    ast_similarity = calculate_ast_similarity(original_file, file)
    semantic_similarity = calculate_semantic_similarity(original_file, file)

    if semantic_similarity > 0.8:
        return 1

    if semantic_similarity < 0.2:
        return 0

    plagiarism_percentage = (semantic_similarity + token_overlap + ast_similarity) / 3.0 * 100
    return plagiarism_percentage

def main():
    print("Iniciando entrenamiento del modelo Clonbusters...")
    
    # Definir rutas correctamente
    current_dir = Path(__file__).parent
    dataset_dir = current_dir.parent / "data" / "IR-Plag-Dataset"
    model_dir = current_dir
    
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
            semantic_similarity = calculate_semantic_similarity(original_file, file)
            
            with open(file, 'r') as f:
                text_samples.append(f.read())

            features.append([token_overlap, ast_similarity, semantic_similarity])
            labels.append(get_label(original_file, file))

    # Entrenar y guardar el vectorizador
    print("Entrenando vectorizador...")
    vectorizer = CountVectorizer()
    vectorizer.fit(text_samples)
    
    # Guardar el vectorizador
    vectorizer_path = model_dir / "vectorizer.joblib"
    print(f"Guardando vectorizador en: {vectorizer_path}")
    dump(vectorizer, vectorizer_path)
    print("¡Vectorizador guardado exitosamente!")

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

    # Guardar el clasificador
    classifier_path = model_dir / "classifier.joblib"
    print(f"Guardando clasificador en: {classifier_path}")
    dump(classifier, classifier_path)
    print("¡Clasificador guardado exitosamente!")

    # Imprimir resumen de archivos guardados
    print("\nResumen de archivos generados:")
    print(f"Vectorizador: {vectorizer_path}")
    print(f"Clasificador: {classifier_path}")

if __name__ == "__main__":
    main()
