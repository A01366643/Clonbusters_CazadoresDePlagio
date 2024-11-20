FROM python:3.8-slim

WORKDIR /app

# Install Java (required for javalang)
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install javalang scikit-learn numpy pandas joblib fastapi uvicorn

# Create necessary directories and copy files
COPY backend/ /app/backend/
COPY model_artifacts/*.joblib /app/backend/model/

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
