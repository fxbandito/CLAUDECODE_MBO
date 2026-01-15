# =============================================================================
# MBO Trading Strategy Analyzer - Dockerfile
# =============================================================================
# Hasznalat:
#   docker build -t mbo-analyzer .
#   docker run -p 8501:8501 mbo-analyzer
#
# GPU tamogatassal:
#   docker run --gpus all -p 8501:8501 mbo-analyzer
# =============================================================================

# CUDA 12.1 alapkep GPU tamogatassal
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Metaadatok
LABEL maintainer="MBO Team"
LABEL description="MBO Trading Strategy Analyzer - Streamlit Web UI"
LABEL version="3.40.0"

# Kornyezeti valtozok
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Rendszer fuggosegek telepitese
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Munkamappa letrehozasa
WORKDIR /app

# Requirements elso (cache optimalizacio)
COPY src/requirements.txt /app/src/requirements.txt

# Python fuggosegek telepitese
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r src/requirements.txt

# PyTorch CUDA tamogatassal
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Alkalmazas masolasa
COPY . /app/

# Szukseges mappak letrehozasa
RUN mkdir -p /app/Data /app/Reports /app/Uploads

# Streamlit konfiguracio (headless mod)
RUN mkdir -p /root/.streamlit
RUN echo '[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /root/.streamlit/config.toml

# Port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Inditas
CMD ["streamlit", "run", "src/web_ui.py"]
