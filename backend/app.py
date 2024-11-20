from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import joblib
import javalang
import os
import numpy as np
from pathlib import Path

app = FastAPI(title="ClonBusters API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://a01366643.github.io",
        "https://clonbusters-backend.onrender.com",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas para los modelos
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

def calculate_semantic_similarity(code1: str, code2: str) -> float:
    """Calcula la similitud semántica usando el modelo entrenado."""
    try:
        if vectorizer is None or classifier is None:
            return 0.0
            
        # Preparar los datos
        combined_code = code1 + " " + code2
        features = vectorizer.transform([combined_code])
        
        # Predecir similitud
        prediction = classifier.predict_proba(features)[0]
        
        # Convertir la predicción a un porcentaje
        similarity = prediction[1] * 100
        return similarity
    except:
        return 0.0

@app.post("/api/analyze")
async def analyze_code(
    original: UploadFile = File(...),
    comparison_files: List[UploadFile] = File(...)
):
    """
    Analiza la similitud entre el código original y los códigos de comparación.
    Retorna métricas de similitud.
    """
    try:
        # Leer contenido del archivo original
        original_content = await original.read()
        original_content = original_content.decode('utf-8')
        
        results = []
        for comparison_file in comparison_files:
            # Leer contenido del archivo de comparación
            comparison_content = await comparison_file.read()
            comparison_content = comparison_content.decode('utf-8')
            
            # Calcular diferentes tipos de similitud
            token_similarity = calculate_token_similarity(original_content, comparison_content)
            ast_similarity = calculate_ast_similarity(original_content, comparison_content)
            semantic_similarity = calculate_semantic_similarity(original_content, comparison_content)
            
            # Calcular puntuación global (puedes ajustar los pesos)
            overall_score = (
                token_similarity * 0.3 +
                ast_similarity * 0.3 +
                semantic_similarity * 0.4
            )
            
            # Determinar si es plagio basado en un umbral
            is_plagiarism = overall_score > 70  # Puedes ajustar este umbral
            
            results.append({
                "filename": comparison_file.filename,
                "token_similarity": round(token_similarity, 1),
                "ast_similarity": round(ast_similarity, 1),
                "semantic_similarity": round(semantic_similarity, 1),
                "overall_score": round(overall_score, 1),
                "is_plagiarism": is_plagiarism
            })
        
        # Por ahora, retornamos solo el primer resultado
        # Podrías modificar esto para manejar múltiples resultados
        return results[0] if results else {
            "error": "No se pudo analizar el código"
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
        "models_loaded": vectorizer is not None and classifier is not None
    }
