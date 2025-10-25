# Usar imagen base con Python 3.11
FROM python:3.11-slim

# Instalar dependencias del sistema necesarias para dlib y opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Instalar CMake 3.27+ (dlib requiere CMake >= 3.5)
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-x86_64.sh \
    && chmod +x cmake-3.27.7-linux-x86_64.sh \
    && ./cmake-3.27.7-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.27.7-linux-x86_64.sh

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements.txt primero (para aprovechar caché de Docker)
COPY requirements.txt .

# Actualizar pip y instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto que usará la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "300"]