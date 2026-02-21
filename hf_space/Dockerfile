FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF, image processing
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for cache efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Expose port
EXPOSE 7860

# Run
CMD ["python", "app.py"]
