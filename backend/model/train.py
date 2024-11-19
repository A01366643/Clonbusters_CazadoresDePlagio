from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import javalang


def load_dataset(base_path):
    """Load all dataset cases from the base directory"""
    return list(base_path.iterdir())


def load_original_file(case_path):
    """Load the original file from a case directory"""
    original_path = case_path / "original"
    return list(original_path.iterdir())[0]


def load_plagiarized_files(case_path):
    """Load all plagiarized files from a case directory"""
    plagiarized_path = case_path / "plagiarized"
    plagiarized_dirs = list(plagiarized_path.iterdir())
    
    plagiarized_files = []
    for directory in plagiarized_dirs:
        for subdir in directory.iterdir():
            plagiarized_files.extend(list(subdir.iterdir()))
    return plagiarized_files


def load_non_plagiarized_files(case_path):
    """Load all non-plagiarized files from a case directory"""
    non_plagiarized_path = case_path / "non-plagiarized"
    non_plagiarized_files = []
    for directory in non_plagiarized_path.iterdir():
        non_plagiarized_files.extend(list(directory.iterdir()))
    return non_plagiarized_files


def extract_features(original_file, comparison_file):
    """Extract features from file pairs"""
    # Token overlap
    def tokenize(file_path):
        with open(file_path, 'r') as file:
            code = file.read()
        tokens = list(javalang.tokenizer.tokenize(code))
        return [token.value for token in tokens]
    
    # Calculate token overlap (Jaccard similarity)
    tokens1 = set(tokenize(original_file))
    tokens2 = set(tokenize(comparison_file))
    token_overlap = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    
    # AST similarity
    def get_ast_nodes(file_path):
        with open(file_path, 'r') as file:
            code = file.read()
        tree = javalang.parse.parse(code)
        return [node.__class__.__name__ for _, node in tree]
    
    nodes1 = get_ast_nodes(original_file)
    nodes2 = get_ast_nodes(comparison_file)
    vectorizer = CountVectorizer().fit([' '.join(nodes1 + nodes2)])
    vec1 = vectorizer.transform([' '.join(nodes1)]).toarray()
    vec2 = vectorizer.transform([' '.join(nodes2)]).toarray()
    ast_similarity = cosine_similarity(vec1, vec2)[0][0]
    
    # Semantic similarity
    with open(original_file, 'r') as f:
        text1 = f.read()
    with open(comparison_file, 'r') as f:
        text2 = f.read()
    
    text_vectorizer = CountVectorizer().fit([text1, text2])
    vec1 = text_vectorizer.transform([text1]).toarray()
    vec2 = text_vectorizer.transform([text2]).toarray()
    semantic_similarity = cosine_similarity(vec1, vec2)[0][0]
    
    return [token_overlap, ast_similarity, semantic_similarity]


def main():
    print('Loading dataset...')
    dataset_path = Path('data/IR-Plag-Dataset')
    cases = load_dataset(dataset_path)
    
    features = []
    labels = []
    
    print('Extracting features...')
    for case in cases:
        original_file = load_original_file(case)
        
        # Process plagiarized files (label 1)
        plagiarized_files = load_plagiarized_files(case)
        for file in plagiarized_files:
            features.append(extract_features(original_file, file))
            labels.append(1)
        
        # Process non-plagiarized files (label 0)
        non_plagiarized_files = load_non_plagiarized_files(case)
        for file in non_plagiarized_files:
            features.append(extract_features(original_file, file))
            labels.append(0)
    
    X = np.array(features)
    y = np.array(labels)
    
    print('Splitting dataset...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print('Training model...')
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    print('Evaluating model...')
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print('Saving model...')
    dump(clf, Path('model/plagiarism-detector-v1.joblib'))


if __name__ == "__main__":
    main()