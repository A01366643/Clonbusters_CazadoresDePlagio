from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import javalang
import numpy as np
from typing import List

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica tus dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo entrenado
model = joblib.load("model_artifacts/model.joblib")  # Ajusta la ruta según tu estructura

@app.post("/api/analyze")
async def analyze_code(original: UploadFile = File(...), comparison_files: List[UploadFile] = File(...)):
    try:
        # Leer el contenido de los archivos
        original_content = await original.read()
        original_content = original_content.decode('utf-8')
        
        results = []
        for comparison_file in comparison_files:
            comparison_content = await comparison_file.read()
            comparison_content = comparison_content.decode('utf-8')
            
            # Calcular similitudes usando tu modelo entrenado
            token_similarity = calculate_token_similarity(original_content, comparison_content)
            ast_similarity = calculate_ast_similarity(original_content, comparison_content)
            semantic_similarity = calculate_semantic_similarity(original_content, comparison_content)
            
            # Calcular puntuación global
            overall_score = (token_similarity + ast_similarity + semantic_similarity) / 3
            
            results.append({
                "token_similarity": token_similarity,
                "ast_similarity": ast_similarity,
                "semantic_similarity": semantic_similarity,
                "overall_score": overall_score,
                "is_plagiarism": overall_score > 70  # Ajusta este umbral según tus necesidades
            })
        
        return results[0]  # Por ahora retornamos solo el primer resultado
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Funciones auxiliares para calcular similitudes
def calculate_token_similarity(code1: str, code2: str) -> float:
    # Implementa tu lógica de comparación de tokens aquí
    # Usando el modelo entrenado
    pass

def calculate_ast_similarity(code1: str, code2: str) -> float:
    # Implementa tu lógica de comparación de AST aquí
    try:
        tree1 = javalang.parse.parse(code1)
        tree2 = javalang.parse.parse(code2)
        # Implementa la comparación de árboles AST
        pass
    except:
        return 0.0

def calculate_semantic_similarity(code1: str, code2: str) -> float:
    # Implementa tu lógica de comparación semántica aquí
    # Usando el modelo entrenado
    pass
