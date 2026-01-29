# Stage 1: Base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SpaCy model to prevent runtime downloads
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application
COPY . .

# Ports for FastAPI (8001) and Streamlit (8501)
EXPOSE 8001
EXPOSE 8501

# The command is handled by docker-compose