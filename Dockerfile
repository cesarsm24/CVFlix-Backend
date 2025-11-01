# ================= BASE =================
FROM python:3.11-slim

# ================= USUARIO =================
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# ================= DIRECTORIO DE TRABAJO =================
WORKDIR /app

# ================= DEPENDENCIAS DEL SISTEMA =================
RUN apt-get update && apt-get install -y \
    build-essential cmake libopenblas-dev liblapack-dev libx11-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ================= COPIAR REQUISITOS =================
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ================= COPIAR EL PROYECTO =================
COPY --chown=user . /app

# ================= INSTALACIÓN ADICIONAL =================
# Aquí ejecutamos tu install.sh para dlib, TensorFlow, etc.
RUN bash install.sh

# ================= COMANDO POR DEFECTO =================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]