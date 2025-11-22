# -------------------------------
# Base Image
# -------------------------------
FROM python:3.10-slim

# -------------------------------
# System Dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Create Work Directory
# -------------------------------
WORKDIR /app

# -------------------------------
# Copy Requirements First
# -------------------------------
COPY requirements.txt .

# -------------------------------
# Install Python Dependencies
# -------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Copy Entire Project
# -------------------------------
COPY . .

# -------------------------------
# Run main.py
# -------------------------------
CMD ["python3", "main.py"]
