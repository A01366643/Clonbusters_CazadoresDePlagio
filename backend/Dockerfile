FROM python:3.8-slim

WORKDIR /app

# Install Java (required for javalang)
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install javalang scikit-learn numpy pandas joblib

# Copy model and application files
COPY model_artifacts/*.joblib /app/model/
COPY model/train.py .
COPY data/ /app/data/

# Command to run the model
CMD ["python", "train.py"]
