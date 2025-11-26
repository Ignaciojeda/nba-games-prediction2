"""
This is a boilerplate pipeline 'regression'
generated using Kedro 1.0.0
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

def prepare_regression_data(data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Prepara los datos para el análisis de regresión
    """
    logger.info("Preparando datos para regresión")
    
    features = parameters["features"]
    targets = parameters["targets"]
    team_identifiers = parameters["team_identifiers"]
    sample_size = parameters["sample_size"]
    random_state = parameters["random_state"]
    
    # VERIFICAR COLUMNAS EXISTENTES
    available_features = [col for col in features if col in data.columns]
    available_targets = {k: v for k, v in targets.items() if v in data.columns}
    available_identifiers = [col for col in team_identifiers if col in data.columns]
    
    columns_needed = available_features + list(available_targets.values()) + available_identifiers
    
    logger.info(f"Features disponibles: {len(available_features)}/{len(features)}")
    logger.info(f"Targets disponibles: {len(available_targets)}/{len(targets)}")
    logger.info(f"Identificadores disponibles: {len(available_identifiers)}/{len(team_identifiers)}")
    
    if not columns_needed:
        logger.error("No hay columnas disponibles para regresión")
        return pd.DataFrame()
    
    df_clean = data[columns_needed].dropna()
    
    # Aplicar muestreo
    if sample_size < len(df_clean):
        df_clean = df_clean.sample(n=sample_size, random_state=random_state, replace=False)
    
    logger.info(f"Datos preparados: {df_clean.shape[0]} muestras")
    if available_identifiers:
        logger.info(f"Equipos únicos: {df_clean[available_identifiers[0]].nunique()}")
    
    return df_clean

def evaluate_models_with_gridsearch(X: pd.DataFrame, y: np.ndarray, target_name: str, 
                                  parameters: Dict) -> Dict[str, Any]:
    """
    Evalúa múltiples modelos con GridSearchCV
    """
    logger.info(f"Entrenando modelos para {target_name}")
    
    gridsearch_params = parameters["gridsearch_params"]
    random_state = parameters["random_state"]
    cv_folds = parameters["cv_folds"]
    test_size = parameters["test_size"]
    
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Definir modelos
    models = {
        'random_forest': RandomForestRegressor(random_state=random_state),
        'gradient_boosting': GradientBoostingRegressor(random_state=random_state),
        'svr': SVR(),
        'ridge': Ridge(random_state=random_state),
        'lasso': Lasso(random_state=random_state),
        'elastic_net': ElasticNet(random_state=random_state),
        'knn': KNeighborsRegressor()
    }
    
    results = {}
    best_score = -np.inf
    best_model = None
    best_model_name = ""
    
    for model_name, model in models.items():
        logger.info(f"Entrenando {model_name} para {target_name}")
        
        try:
            # Crear pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # GridSearch
            grid_search = GridSearchCV(
                pipeline,
                gridsearch_params[model_name],
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            # Entrenar
            grid_search.fit(X_train, y_train)
            
            # Predecir
            y_pred = grid_search.predict(X_test)
            
            # Métricas
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            results[model_name] = {
                'best_params': grid_search.best_params_,
                'cv_score': grid_search.best_score_,
                'test_r2': r2,
                'test_mae': mae,
                'test_mse': mse,
                'model': grid_search.best_estimator_
            }
            
            logger.info(f"{model_name} - CV R²: {grid_search.best_score_:.4f}, Test R²: {r2:.4f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = grid_search.best_estimator_
                best_model_name = model_name
                
        except Exception as e:
            logger.error(f"Error en {model_name} para {target_name}: {str(e)}")
            continue
    
    logger.info(f"Mejor modelo para {target_name}: {best_model_name} (R²: {best_score:.4f})")
    
    return {
        'results': results,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_score': best_score
    }

def train_local_strength_model(regression_data: pd.DataFrame, parameters: Dict) -> Tuple[Any, Dict]:
    """
    Entrena modelo para fuerza local
    """
    features = parameters["features"]
    target_col = parameters["targets"]["local_strength"]
    
    X = regression_data[features]
    y = regression_data[target_col].values
    
    results = evaluate_models_with_gridsearch(X, y, "local_strength", parameters)
    
    return results["best_model"], results

def train_away_weakness_model(regression_data: pd.DataFrame, parameters: Dict) -> Tuple[Any, Dict]:
    """
    Entrena modelo para debilidad visitante
    """
    features = parameters["features"]
    target_col = parameters["targets"]["away_weakness"]
    
    X = regression_data[features]
    y = regression_data[target_col].values
    
    results = evaluate_models_with_gridsearch(X, y, "away_weakness", parameters)
    
    return results["best_model"], results

def train_point_differential_model(regression_data: pd.DataFrame, parameters: Dict) -> Tuple[Any, Dict]:
    """
    Entrena modelo para diferencial de puntos
    """
    features = parameters["features"]
    target_col = parameters["targets"]["point_differential"]
    
    X = regression_data[features]
    y = regression_data[target_col].values
    
    results = evaluate_models_with_gridsearch(X, y, "point_differential", parameters)
    
    return results["best_model"], results

def create_model_report(local_results: Dict, away_results: Dict, 
                       diff_results: Dict, parameters: Dict) -> Dict:
    """
    Crea reporte consolidado de todos los modelos
    """
    logger.info("Creando reporte de modelos")
    
    report = {
        "summary": {
            "total_targets": 3,
            "best_models": {
                "local_strength": local_results["best_model_name"],
                "away_weakness": away_results["best_model_name"],
                "point_differential": diff_results["best_model_name"]
            },
            "best_scores": {
                "local_strength": local_results["best_score"],
                "away_weakness": away_results["best_score"],
                "point_differential": diff_results["best_score"]
            }
        },
        "detailed_results": {
            "local_strength": {
                model: {k: v for k, v in metrics.items() if k != 'model'} 
                for model, metrics in local_results["results"].items()
            },
            "away_weakness": {
                model: {k: v for k, v in metrics.items() if k != 'model'} 
                for model, metrics in away_results["results"].items()
            },
            "point_differential": {
                model: {k: v for k, v in metrics.items() if k != 'model'} 
                for model, metrics in diff_results["results"].items()
            }
        },
        "parameters_used": parameters
    }
    
    logger.info("Reporte de modelos creado exitosamente")
    return report

def enhance_regression_features_with_clusters(regression_data: pd.DataFrame,
                                            clustering_results: Dict,
                                            dimensionality_results: Dict) -> pd.DataFrame:
    """Mejorar features de regresión con información de clusters y PCA."""
    
    enhanced_data = regression_data.copy()
    
    # Obtener el mejor algoritmo de clustering
    best_algo = clustering_results.get('best_algorithm', 'kmeans')
    cluster_labels = clustering_results[best_algo]['labels']
    
    # Agregar cluster como feature
    enhanced_data['TEAM_CLUSTER'] = cluster_labels[:len(enhanced_data)]
    
    # Agregar métricas de cluster
    cluster_stats = _calculate_cluster_statistics(clustering_results, regression_data)
    enhanced_data = enhanced_data.merge(cluster_stats, on='TEAM_CLUSTER', how='left')
    
    return enhanced_data

def _calculate_cluster_statistics(clustering_results: Dict, regression_data: pd.DataFrame) -> pd.DataFrame:
    """Calcular estadísticas por cluster para usar como features."""
    # Implementar cálculo de estadísticas por cluster
    pass

def enhance_regression_with_clusters(regression_data: pd.DataFrame, 
                                   clustering_results: Dict,
                                   team_features_base: pd.DataFrame) -> pd.DataFrame:
    """Mejorar datos de regresión con información de clusters."""
    
    enhanced_data = regression_data.copy()
    
    # Obtener clusters del mejor algoritmo
    best_algo = clustering_results.get('best_algorithm', 'kmeans')
    cluster_labels = clustering_results[best_algo]['labels']
    
    # Agregar cluster como feature
    enhanced_data['TEAM_CLUSTER'] = cluster_labels[:len(enhanced_data)]
    
    return enhanced_data