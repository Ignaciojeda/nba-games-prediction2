🏀 Pipeline de Análisis y Predicción NBA con Kedro
📋 Descripción del Proyecto
Este proyecto analiza datos de la NBA para identificar patrones de rendimiento de equipos y genera modelos de machine learning para predecir resultados de partidos.
Utiliza el framework Kedro para crear pipelines reproducibles y modulares, integrando MLflow para experimentación, DVC para control de versiones de datos, Airflow para orquestación, y Docker para containerización.

🎯 Objetivos Principales
📌 Identificar qué equipos son mejores jugando en casa

📌 Analizar qué equipos son peores como visitantes

📌 Predecir si el equipo local ganará (Clasificación)

📌 Predecir métricas de rendimiento (Regresión):

local_strength: Fuerza del equipo local

away_weakness: Debilidad del equipo visitante

point_differential: Diferencia de puntos

home_advantage: Ventaja de jugar en casa

📌 Crear pipelines modulares y reproducibles

🤖 Modelos de Machine Learning
🎯 Modelo de Clasificación
Objetivo: Predecir si el equipo local ganará el partido

Características principales:

Tasas de victoria históricas (local/visitante)

Rendimiento reciente (últimos 10 partidos)

Métricas de fuerza del equipo

Historial de enfrentamientos

Métricas de evaluación:

Precisión (Accuracy)

Precisión (Precision)

Recall

Puntuación F1

ROC-AUC

📈 Modelo de Regresión
Objetivo: Predecir métricas continuas de rendimiento

Variables objetivo:

local_strength: Puntuación de fuerza del equipo local (0-100)

away_weakness: Índice de debilidad del visitante (0-100)

point_differential: Diferencia esperada de puntos (±)

home_advantage: Ventaja por jugar en casa (0-20)

Métricas de evaluación:

RMSE (Error Cuadrático Medio)

MAE (Error Absoluto Medio)

Puntuación R²

🚀 Inicio Rápido
Prerrequisitos
Python 3.8+

Docker y Docker Compose

Git

Opción 1: Docker (Recomendada)

# Clonar repositorio
git clone <url-del-repositorio>
cd nba-analysis

# Construir y ejecutar con Docker
docker-compose up --build

# Acceder a los servicios:
# MLflow: http://localhost:5000
# Airflow: http://localhost:8080 (usuario: airflow, contraseña: airflow)
# Kedro Viz: http://localhost:4141

Opción 2: Desarrollo Local

# Clonar y configurar
git clone <url-del-repositorio>
cd nba-analysis

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
pip install -r requirements_ml.txt

# Configurar datos
mkdir -p data/01_raw
# Colocar datasets en data/01_raw/

# Inicializar DVC
dvc init

# Ejecutar pipeline
kedro run

📊 Datasets Utilizados
El proyecto utiliza 3 datasets principales de Kaggle (NBA Games):

Dataset	Descripción
games.csv	Resultados y estadísticas de partidos
games_details.csv	Estadísticas por jugador por partido
teams.csv	Información de los equipos
Fuente: Kaggle - NBA Games Dataset

🐳 Configuración de Docker
Construir y Ejecutar