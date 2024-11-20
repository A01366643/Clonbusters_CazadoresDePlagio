from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import javalang
from joblib import dump

# ... (mantener todas las funciones anteriores igual hasta main()) ...

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
