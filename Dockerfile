FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar proyecto (excluyendo lo que no se necesita)
COPY src/ ./src/
COPY conf/ ./conf/
COPY data/ ./data/
COPY notebooks/ ./notebooks/
COPY pyproject.toml .

# Exponer puerto para Kedro Viz
EXPOSE 4141

# Comando por defecto
CMD ["kedro", "run"]