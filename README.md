🏀 Pipeline de Análisis y Predicción NBA con Kedro
📋 Descripción del Proyecto
Este proyecto analiza datos de la NBA para identificar patrones de rendimiento de equipos y genera modelos de machine learning para predecir resultados de partidos.
Utiliza el framework Kedro para crear pipelines reproducibles y modulares, integrando DVC para control de versiones de datos, Airflow para orquestación, y Docker para containerización.

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

Algoritmos implementados:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVC)

Gaussian Naive Bayes

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


Integración de Clustering 

Clasificación: 36 nuevas features agregadas (de 17 a 53 features totales)

Regresión: 21 nuevas features agregadas (de 17 a 38 features totales)

Clusters encontrados: 2 clusters con silhouette score de 0.6237

📊 Rendimiento de Modelos:

Clasificación:

Mejor modelo: Logistic Regression

Accuracy: 99.61% ✅

F1-Score: 99.67% ✅

AUC-ROC: 99.89% ✅

Mejora vs baseline: +69.7% 📈

Regresión:

Local Strength (PTS_home): R² = 0.7797 (Ridge)

Away Weakness (PTS_away): R² = 0.7769 (Ridge)

Point Differential: R² = 0.7368 (Ridge)

🎯 Features de Clustering Agregadas:

Para Clasificación (36 features):

HOME_CLUSTER_0, HOME_CLUSTER_1 (one-hot encoding)

AWAY_CLUSTER_0, AWAY_CLUSTER_1 (one-hot encoding)

SAME_CLUSTER, CLUSTER_DIFF

Estadísticas por cluster para ambos equipos

Para Regresión (21 features):

CLUSTER_0, CLUSTER_1 (one-hot encoding)

Estadísticas de cluster

Distancias a centroides

📈 Impacto de la Integración:

Mejor comprensión de los patrones de equipos

Features adicionales que capturan relaciones entre equipos

Mejor rendimiento predictivo en ambos tipos de modelos

Cumplimiento completo de los requisitos de integración supervisado + no supervisado

🚀 Inicio Rápido
Prerrequisitos
Python 3.8+

Docker y Docker Compose

Git

Opción 1: Docker (Recomendada)

```# Clonar repositorio
git clone <url-del-repositorio>
cd nba-analysis

# Construir y ejecutar con Docker
docker-compose up --build

# Acceder a los servicios:
# Airflow: http://localhost:8080 (usuario: airflow, contraseña: airflow)
# Kedro Viz: http://localhost:4141


Opción 2: Desarrollo Local
```
# Clonar y configurar
git clone <url-del-repositorio>
cd nba-analysis

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar datos
mkdir -p data/01_raw
# Colocar datasets en data/01_raw/

# Inicializar DVC
dvc init

# Ejecutar pipeline
kedro run
```

Interfaz Web con Streamlit
El proyecto incluye una interfaz web interactiva desarrollada con Streamlit...

📊 Datasets Utilizados
El proyecto utiliza 3 datasets principales de Kaggle (NBA Games):

Dataset	Descripción
games.csv	Resultados y estadísticas de partidos
games_details.csv	Estadísticas por jugador por partido
teams.csv	Información de los equipos
Fuente: Kaggle - NBA Games Dataset

🐳 Configuración de Docker

Construir y Ejecutar
```
# Construir imágenes
docker-compose build

# Ejecutar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

Resumen de Servicios
Servicio	Puerto	Propósito
Pipeline Kedro	-	Procesamiento de datos y entrenamiento de modelos
Airflow	8080	Orquestación de pipelines
Kedro Viz	4141	Visualización de pipelines
Streamlit	8501	Interfaz web interactiva para predicciones y análisis


Comandos Útiles de Docker

```

# Ejecutar comandos específicos
docker-compose run app kedro run
docker-compose run app kedro test

# Acceder a la shell del contenedor
docker-compose exec app bash

# Ver logs de servicios específicos
docker-compose logs -f app

```

🌪️ Orquestación con Airflow

Iniciar Airflow
```
# Usando Docker Compose
docker-compose -f docker-compose-airflow.yml up -d

# Acceder a la interfaz web: http://localhost:8080
# Usuario: airflow, Contraseña: airflow
```
DAG Principal - Pipeline NBA
El DAG de Airflow ejecuta automáticamente:

1.Preprocesamiento de datos y validación

2.Entrenamiento de modelos (clasificación y regresión)

3.Evaluación de modelos y cálculo de métricas

4.Generación de predicciones y reportes

Comandos de Airflow
```

# Listar DAGs
docker-compose -f docker-compose-airflow.yml exec webserver airflow dags list

# Ejecutar DAG manualmente
docker-compose -f docker-compose-airflow.yml exec webserver airflow dags trigger nba_pipeline

# Ver estado de tareas
docker-compose -f docker-compose-airflow.yml exec webserver airflow tasks list nba_pipeline
```
📊 DVC (Control de Versiones de Datos)

Configuración Inicial
```
# Inicializar DVC
dvc init

# Configurar almacenamiento remoto (ejemplo local)
dvc remote add -d myremote /ruta/al/almacenamiento

# Para almacenamiento en la nube (ejemplo AWS S3):
dvc remote add -d myremote s3://mi-bucket/nba-data

Flujo de Trabajo con DVC

# Añadir datasets al seguimiento de DVC
dvc add data/01_raw/games.csv
dvc add data/01_raw/games_details.csv
dvc add data/01_raw/teams.csv

# Añadir modelos al seguimiento
dvc add models/classification_model.pkl
dvc add models/regression_model.pkl

# Hacer commit en Git
git add .dvc data/01_raw/.gitignore models/.gitignore
git commit -m "Añadir datasets y modelos a DVC"
```
Flujo de Trabajo Diario
```
# Obtener datos más recientes
dvc pull

# Ejecutar pipeline
kedro run

# Añadir nuevos resultados a DVC
dvc add data/06_models/predictions.csv
dvc add models/

# Subir cambios al almacenamiento remoto
dvc push

# Hacer commit de cambios de metadatos
git add .dvc
git commit -m "Actualizar modelos y predicciones"
git push
```

Reproducir Experimentos
```
# Reproducir pipeline específico
dvc repro pipelines/data_processing

# Verificar estado
dvc status

# Comparar métricas entre versiones
dvc metrics diff
```
🏃‍♂️ Ejecución del Proyecto

Pipeline Completo
```
# Usando Kedro directamente
kedro run

# Usando Docker
docker-compose run app kedro run

# Usando Airflow
# Activar el DAG nba_pipeline en la interfaz de Airflow
```
Pipelines Específicos
```
# Solo procesamiento de datos
kedro run --pipeline=data_engineering


# Solo entrenamiento de clasificación
kedro run --pipeline=classification

# Solo entrenamiento de regresión
kedro run --pipeline=regression
```
Visualización y Análisis
```
# Visualización del pipeline
kedro viz
# Acceder: http://localhost:4141

# Notebooks Jupyter con contexto de Kedro
kedro jupyter notebook

# Notebook de análisis específico
kedro jupyter notebook --notebook-path notebooks/04_model_analysis.ipynb
```
🔧 Comandos Útiles
Información del Proyecto
```
# Resumen del proyecto
kedro info

# Listar datasets disponibles
kedro catalog list

# Ejecutar tests
kedro test

# Crear nuevo pipeline
kedro pipeline create <nombre_pipeline>
```
Gestión de Servicios
```
# Iniciar todos los servicios
docker-compose up -d

# Detener todos los servicios
docker-compose down

# Verificar estado de servicios
docker-compose ps

# Ver logs de servicios
docker-compose logs -f <nombre_servicio>
```
Gestión de Datos y Modelos
```
# Verificar estado de DVC
dvc status

# Obtener datos más recientes
dvc pull

# Subir cambios al remoto
dvc push

# Reproducir pipeline
dvc repro
```
