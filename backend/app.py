from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import joblib
import javalang
import os
import numpy as np
from pathlib import Path
from collections import Counter
import math

app = FastAPI(title="ClonBusters API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://a01366643.github.io",
        "http://localhost:5173",
        "http://localhost:8000",
        "https://clonbusters-backend.onrender.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

""" # Rutas para los modelos
MODEL_DIR = Path("/app/backend/model")
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIR / "classifier.joblib"

# Cargar los modelos entrenados
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
except Exception as e:
    print(f"Error cargando los modelos: {e}")
    # En producción, podrías querer manejar esto de manera diferente
    vectorizer = None
    classifier = None
 """
def extract_features(java_code: str) -> dict:
    """Extrae características del código Java."""
    try:
        # Parsear el código Java
        tree = javalang.parse.parse(java_code)
        
        # Extraer características básicas
        features = {
            'n_methods': 0,
            'n_classes': 0,
            'n_loops': 0,
            'n_variables': 0,
            'code_length': len(java_code),
        }
        
        # Contar elementos
        for path, node in tree:
            if isinstance(node, javalang.tree.MethodDeclaration):
                features['n_methods'] += 1
            elif isinstance(node, javalang.tree.ClassDeclaration):
                features['n_classes'] += 1
            elif isinstance(node, (javalang.tree.ForStatement, javalang.tree.WhileStatement)):
                features['n_loops'] += 1
            elif isinstance(node, javalang.tree.LocalVariableDeclaration):
                features['n_variables'] += 1
                
        return features
    except:
        return {
            'n_methods': 0,
            'n_classes': 0,
            'n_loops': 0,
            'n_variables': 0,
            'code_length': len(java_code),
        }

def calculate_token_similarity(code1: str, code2: str) -> float:
    """Calcula la similitud basada en tokens."""
    try:
        tokens1 = list(javalang.tokenizer.tokenize(code1))
        tokens2 = list(javalang.tokenizer.tokenize(code2))
        
        # Convertir tokens a strings para comparación
        tokens1_str = [str(t.value) for t in tokens1]
        tokens2_str = [str(t.value) for t in tokens2]
        
        # Calcular intersección
        common_tokens = set(tokens1_str) & set(tokens2_str)
        
        # Calcular similitud Jaccard
        similarity = len(common_tokens) / (len(set(tokens1_str) | set(tokens2_str)))
        
        return similarity * 100
    except:
        return 0.0

def calculate_ast_similarity(code1: str, code2: str) -> float:
    """Calcula la similitud basada en AST."""
    try:
        features1 = extract_features(code1)
        features2 = extract_features(code2)
        
        # Calcular similitud basada en características
        total_diff = 0
        total_features = len(features1)
        
        for key in features1:
            max_val = max(features1[key], features2[key])
            if max_val == 0:
                continue
            diff = abs(features1[key] - features2[key]) / max_val
            total_diff += diff
        
        similarity = (1 - (total_diff / total_features)) * 100
        return similarity
    except:
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


def extract_file_features(tree):
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

def calculate_semantic_similarity(code1: str, code2: str) -> float:
    """Calcula la similitud semántica usando el modelo entrenado."""
    try:
        tree1 = javalang.parse.parse(code1)
        tree2 = javalang.parse.parse(code2)

        features_original_file = extract_file_features(tree1)
        features_file = extract_file_features(tree2)

        feature_similarity_var = np.mean([
            np.abs(a - b) for a, b in zip(features_original_file, features_file)
            ])

        similarity = (abs(1 - feature_similarity_var)) * 100

        return similarity
    except:
        return 0.0

@app.post("/api/analyze")
async def analyze_code(
    original: UploadFile = File(...),
    comparison_file: UploadFile = File(...)  # Cambiado de comparison_files a comparison_file
):
    """
    Analiza la similitud entre el código original y el código de comparación.
    Retorna métricas de similitud.
    """
    try:
        # Leer contenido del archivo original
        original_content = await original.read()
        original_content = original_content.decode('utf-8')
        
        # Leer contenido del archivo de comparación
        comparison_content = await comparison_file.read()
        comparison_content = comparison_content.decode('utf-8')
        
        # Calcular diferentes tipos de similitud
        token_similarity = calculate_token_similarity(original_content, comparison_content)
        ast_similarity = calculate_ast_similarity(original_content, comparison_content)
        semantic_similarity = calculate_semantic_similarity(original_content, comparison_content)
        
        # Calcular puntuación global 
        overall_score = (
            ast_similarity * 0.4 +
            token_similarity * 0.3 +
            semantic_similarity * 0.3
        )
        
        # Determinar si es plagio basado en un umbral
        is_plagiarism = overall_score > 70  
        
        return {
            "filename": comparison_file.filename,
            "token_similarity": round(token_similarity, 1),
            "ast_similarity": round(ast_similarity, 1),
            "semantic_similarity": round(semantic_similarity, 1),
            "overall_score": 1,
            "is_plagiarism": is_plagiarism
        }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al analizar el código: {str(e)}"
        )

# Ruta de verificación de salud
@app.get("/health")
async def health_check():
    """Verifica el estado de la API y los modelos."""
    return {
        "status": "ok",
        #"models_loaded": vectorizer is not None and classifier is not None
    }