# Base con soporte amplio para PyTorch/Ultralytics en Linux
FROM python:3.11-slim

# Paquetes del sistema necesarios para OpenCV y FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de la app
WORKDIR /app

# Copiar requirements y instalar dependencias
COPY Modelo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el código de la app
COPY Modelo/ /app/

# Crear directorios de datos
RUN mkdir -p /app/outputs /app/uploads

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# Exponer el puerto
EXPOSE 8000

# Comando de arranque
CMD ["bash", "-lc", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}"]