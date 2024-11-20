FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copiar requerimientos e instalar dependencias
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Crear directorio para modelos
RUN mkdir -p /app/backend/model

# Copiar los archivos del modelo
COPY backend/model/*.joblib /app/backend/model/

# Copiar el resto del código
COPY backend/ /app/backend/

# Puerto por defecto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
