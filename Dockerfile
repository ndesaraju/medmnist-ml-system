# Use a modern Python version
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for PIL, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip safely inside container
RUN python -m pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command (can override)
CMD ["bash"]
# CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]