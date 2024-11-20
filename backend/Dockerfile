FROM python:3.8-slim
WORKDIR /app

# Install Java (required for javalang)
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN pip install javalang scikit-learn numpy pandas joblib

# Create necessary directories
RUN mkdir -p /app/backend/model /app/backend/data

# Copy model and application files
COPY model_artifacts/*.joblib /app/backend/model/
COPY backend/model/train.py /app/backend/model/
COPY backend/data/ /app/backend/data/

# Command to run the model
CMD ["python", "/app/backend/model/train.py"]

# Instalar FastAPI y uvicorn
RUN pip install fastapi uvicorn

# Copiar el archivo app.py
COPY backend/app.py /app/backend/

# Cambiar el comando para ejecutar la API
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
