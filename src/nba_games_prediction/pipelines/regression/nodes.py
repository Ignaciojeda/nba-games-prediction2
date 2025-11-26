"""
Nodes actualizados para pipeline de regresión con integración de clustering
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

def prepare_regression_data_with_clusters(
    data: pd.DataFrame, 
    clustering_results: Dict,
    team_features_base: pd.DataFrame,  # NUEVO INPUT
    parameters: Dict
) -> pd.DataFrame:
    """
    Prepara datos para regresión integrando información de clusters
    """
    logger.info("Preparando datos para regresión con clusters")
    
    features = parameters["features"]
    targets = parameters["targets"]
    team_identifiers = parameters["team_identifiers"]
    sample_size = parameters["sample_size"]
    random_state = parameters["random_state"]
    
    # Verificar columnas existentes
    available_features = [col for col in features if col in data.columns]
    available_targets = {k: v for k, v in targets.items() if v in data.columns}
    available_identifiers = [col for col in team_identifiers if col in data.columns]
    
    columns_needed = available_features + list(available_targets.values()) + available_identifiers
    
    logger.info(f"Features disponibles: {len(available_features)}/{len(features)}")
    
    if not columns_needed:
        logger.error("No hay columnas disponibles para regresión")
        return pd.DataFrame()
    
    df_clean = data[columns_needed].dropna()
    
    # INTEGRACIÓN CON CLUSTERING - PASAMOS team_features_base
    df_enhanced = _enhance_with_clustering_features(df_clean, clustering_results, team_features_base, available_identifiers)
    
    # Aplicar muestreo
    if sample_size < len(df_enhanced):
        df_enhanced = df_enhanced.sample(n=sample_size, random_state=random_state, replace=False)
    
    logger.info(f"Datos preparados con clusters: {df_enhanced.shape[0]} muestras, {df_enhanced.shape[1]} features")
    logger.info(f"Nuevas features: {[col for col in df_enhanced.columns if col not in columns_needed]}")
    
    return df_enhanced

def _enhance_with_clustering_features(
    data: pd.DataFrame, 
    clustering_results: Dict,
    team_features_base: pd.DataFrame,
    team_identifiers: list
) -> pd.DataFrame:
    """
    Mejora los datos con features derivadas del clustering
    """
    enhanced_data = data.copy()
    
    # DEBUG: Ver todas las columnas disponibles
    all_columns = enhanced_data.columns.tolist()
    logger.info(f"Todas las columnas disponibles en regresión: {all_columns}")
    
    # Obtener el mejor algoritmo de clustering
    best_algo = clustering_results.get('best_algorithm', 'kmeans')
    best_results = clustering_results[best_algo]
    cluster_labels = best_results['labels']
    
    # Crear mapeo de equipo a cluster usando team_features_base
    team_cluster_map = {}
    if 'TEAM_ID' in team_features_base.columns:
        # Asumimos que team_features_base está en el mismo orden que los clusters
        for i, team_id in enumerate(team_features_base['TEAM_ID'].values):
            if i < len(cluster_labels):
                team_cluster_map[team_id] = cluster_labels[i]
    
    logger.info(f"Mapeo de equipos a clusters creado: {len(team_cluster_map)} equipos")
    logger.info(f"Ejemplos de mapeo: {list(team_cluster_map.items())[:5]}")
    
    # Agregar cluster como feature
    team_id_col = None
    
    # ESTRATEGIA 1: Usar team_identifiers si están disponibles
    if team_identifiers:
        for identifier in team_identifiers:
            if identifier in enhanced_data.columns:
                team_id_col = identifier
                break
    
    # ESTRATEGIA 2: Buscar columnas de equipo automáticamente
    if not team_id_col:
        potential_team_cols = [col for col in enhanced_data.columns if any(keyword in col.upper() for keyword in ['TEAM', 'HOME', 'AWAY', 'ID'])]
        if potential_team_cols:
            team_id_col = potential_team_cols[0]
            logger.info(f"Columna de equipo seleccionada automáticamente: {team_id_col}")
    
    logger.info(f"Columna de equipo final: {team_id_col}")
    
    if team_id_col and team_cluster_map:
        # Verificar que la columna existe y tiene datos
        logger.info(f"Valores únicos en {team_id_col}: {enhanced_data[team_id_col].nunique()}")
        
        # Agregar cluster del equipo
        enhanced_data['TEAM_CLUSTER'] = enhanced_data[team_id_col].map(team_cluster_map)
        
        # Llenar NaN con -1 (para equipos no encontrados)
        enhanced_data['TEAM_CLUSTER'] = enhanced_data['TEAM_CLUSTER'].fillna(-1)
        
        logger.info(f"Clusters asignados: {enhanced_data['TEAM_CLUSTER'].value_counts().to_dict()}")
        
        # Codificar clusters como variables dummy
        cluster_dummies = pd.get_dummies(enhanced_data['TEAM_CLUSTER'], prefix='CLUSTER')
        enhanced_data = pd.concat([enhanced_data, cluster_dummies], axis=1)
        
        # Agregar características de cluster (estadísticas)
        cluster_stats = _calculate_cluster_statistics_for_regression(clustering_results, team_features_base)
        if not cluster_stats.empty:
            enhanced_data = enhanced_data.merge(
                cluster_stats.add_prefix('CLUSTER_STATS_'), 
                left_on='TEAM_CLUSTER', 
                right_index=True, 
                how='left'
            )
        
        # Agregar métricas de distancia a centroides si están disponibles
        if 'centers' in best_results:
            enhanced_data = _add_cluster_distance_features(enhanced_data, best_results, team_cluster_map, team_id_col)
    else:
        # ESTRATEGIA DE FALLBACK
        logger.warning("No se encontró columna de equipo adecuada. Usando estrategia de fallback.")
        if len(cluster_labels) >= len(enhanced_data):
            enhanced_data['TEAM_CLUSTER'] = cluster_labels[:len(enhanced_data)]
            cluster_dummies = pd.get_dummies(enhanced_data['TEAM_CLUSTER'], prefix='CLUSTER')
            enhanced_data = pd.concat([enhanced_data, cluster_dummies], axis=1)
    
    # Llenar valores NaN
    enhanced_data = enhanced_data.fillna(0)
    
    # Eliminar columna temporal si existe
    if 'TEAM_CLUSTER' in enhanced_data.columns:
        enhanced_data = enhanced_data.drop('TEAM_CLUSTER', axis=1)
    
    # DEBUG: Mostrar nuevas features agregadas
    new_features = [col for col in enhanced_data.columns if col not in all_columns]
    logger.info(f"✅ Nuevas features agregadas en regresión: {new_features}")
    logger.info(f"✅ Total de nuevas features: {len(new_features)}")
    
    return enhanced_data

def _calculate_cluster_statistics_for_regression(clustering_results: Dict, team_features_base: pd.DataFrame) -> pd.DataFrame:
    """Calcular estadísticas por cluster para usar como features"""
    best_algo = clustering_results.get('best_algorithm', 'kmeans')
    cluster_labels = clustering_results[best_algo]['labels']
    
    # Crear DataFrame con clusters
    team_clusters = team_features_base.copy()
    team_clusters['CLUSTER'] = cluster_labels[:len(team_features_base)]
    
    # Seleccionar columnas numéricas para estadísticas
    numeric_cols = team_clusters.select_dtypes(include=[np.number]).columns
    # Excluir columnas no relevantes
    exclude_cols = ['TEAM_ID', 'SEASON', 'CLUSTER']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Tomar las 5 columnas más importantes
    if len(numeric_cols) > 5:
        # Usar varianza para seleccionar las más importantes
        variances = team_clusters[numeric_cols].var()
        numeric_cols = variances.nlargest(5).index.tolist()
    
    cluster_stats = {}
    unique_clusters = team_clusters['CLUSTER'].unique()
    
    for cluster_id in unique_clusters:
        if cluster_id != -1:  # Ignorar noise
            cluster_data = team_clusters[team_clusters['CLUSTER'] == cluster_id]
            if len(cluster_data) > 0:
                stats = {}
                for col in numeric_cols:
                    if col in cluster_data.columns:
                        stats[f'{col}_MEAN'] = cluster_data[col].mean()
                        stats[f'{col}_STD'] = cluster_data[col].std()
                        stats[f'{col}_MEDIAN'] = cluster_data[col].median()
                
                cluster_stats[cluster_id] = stats
    
    return pd.DataFrame(cluster_stats).T

def _add_cluster_distance_features(enhanced_data: pd.DataFrame, best_results: Dict, 
                                 team_cluster_map: Dict, team_id_col: str) -> pd.DataFrame:
    """Agregar features de distancia a centroides"""
    data_temp = enhanced_data.copy()
    
    centers = best_results['centers']
    
    # Para cada equipo, agregar features de su cluster
    for team_id, cluster_id in team_cluster_map.items():
        if cluster_id != -1 and cluster_id < len(centers):
            mask = data_temp[team_id_col] == team_id
            # Feature binaria indicando pertenencia al cluster
            data_temp.loc[mask, f'IN_CLUSTER_{cluster_id}'] = 1
            # Feature de distancia (simplificada - 0 para equipos en el cluster)
            data_temp.loc[mask, f'CLUSTER_DIST_{cluster_id}'] = 0
    
    # Llenar NaN
    cluster_cols = [col for col in data_temp.columns if 'IN_CLUSTER_' in col or 'CLUSTER_DIST_' in col]
    data_temp[cluster_cols] = data_temp[cluster_cols].fillna(0)
    
    return data_temp

def evaluate_models_with_clusters(
    X: pd.DataFrame, 
    y: np.ndarray, 
    target_name: str,
    parameters: Dict
) -> Dict[str, Any]:
    """
    Evalúa modelos con datos que incluyen features de clustering
    """
    logger.info(f"Entrenando modelos con clusters para {target_name}")
    
    gridsearch_params = parameters["gridsearch_params"]
    random_state = parameters["random_state"]
    cv_folds = parameters["cv_folds"]
    test_size = parameters["test_size"]
    
    # Identificar features de clustering
    cluster_features = [col for col in X.columns if 'CLUSTER' in col]
    logger.info(f"Features de clustering utilizadas: {len(cluster_features)}")
    if cluster_features:
        logger.info(f"Ejemplos: {cluster_features[:5]}...")
    
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Features totales: {X_train.shape[1]}, Features clustering: {len(cluster_features)}")
    
    # Definir modelos (actualizados para manejar más features)
    models = {
        'random_forest': RandomForestRegressor(random_state=random_state, n_estimators=100),
        'gradient_boosting': GradientBoostingRegressor(random_state=random_state, n_estimators=100),
        'ridge': Ridge(random_state=random_state),
        'lasso': Lasso(random_state=random_state),
        'elastic_net': ElasticNet(random_state=random_state),
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
                'model': grid_search.best_estimator_,
                'feature_importance': _get_feature_importance(grid_search.best_estimator_, X.columns)
            }
            
            logger.info(f"{model_name} - CV R²: {grid_search.best_score_:.4f}, Test R²: {r2:.4f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = grid_search.best_estimator_
                best_model_name = model_name
                
        except Exception as e:
            logger.error(f"Error en {model_name} para {target_name}: {str(e)}")
            continue
    
    # Análisis de importancia de features de clustering
    cluster_importance = _analyze_cluster_feature_importance(results, cluster_features)
    results['cluster_analysis'] = cluster_importance
    
    logger.info(f"Mejor modelo para {target_name}: {best_model_name} (R²: {best_score:.4f})")
    
    return {
        'results': results,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_score': best_score,
        'cluster_features_used': cluster_features
    }

def _get_feature_importance(model, feature_names):
    """Obtener importancia de features si el modelo lo soporta"""
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        return dict(zip(feature_names, importances))
    elif hasattr(model.named_steps['model'], 'coef_'):
        coef = model.named_steps['model'].coef_
        if len(coef.shape) > 1:
            coef = coef[0]  # Para multi-class
        return dict(zip(feature_names, np.abs(coef)))
    return {}

def _analyze_cluster_feature_importance(results: Dict, cluster_features: list) -> Dict:
    """Analiza la importancia de las features de clustering"""
    cluster_analysis = {}
    
    for model_name, model_info in results.items():
        if 'feature_importance' in model_info and model_info['feature_importance']:
            importances = model_info['feature_importance']
            cluster_importances = {feat: importances.get(feat, 0) for feat in cluster_features}
            total_cluster_importance = sum(cluster_importances.values())
            total_importance = sum(importances.values()) if importances else 1
            
            cluster_analysis[model_name] = {
                'cluster_features_importance': cluster_importances,
                'total_cluster_importance': total_cluster_importance,
                'percentage_cluster_importance': (total_cluster_importance / total_importance * 100) if total_importance > 0 else 0
            }
    
    return cluster_analysis

# Funciones de entrenamiento específicas
def train_local_strength_model(regression_data: pd.DataFrame, parameters: Dict) -> Tuple[Any, Dict]:
    """Entrena modelo para fuerza local con clusters"""
    features = [col for col in regression_data.columns if col not in parameters["targets"].values()]
    target_col = parameters["targets"]["local_strength"]
    
    X = regression_data[features]
    y = regression_data[target_col].values
    
    results = evaluate_models_with_clusters(X, y, "local_strength", parameters)
    return results["best_model"], results

def train_away_weakness_model(regression_data: pd.DataFrame, parameters: Dict) -> Tuple[Any, Dict]:
    """Entrena modelo para debilidad visitante con clusters"""
    features = [col for col in regression_data.columns if col not in parameters["targets"].values()]
    target_col = parameters["targets"]["away_weakness"]
    
    X = regression_data[features]
    y = regression_data[target_col].values
    
    results = evaluate_models_with_clusters(X, y, "away_weakness", parameters)
    return results["best_model"], results

def train_point_differential_model(regression_data: pd.DataFrame, parameters: Dict) -> Tuple[Any, Dict]:
    """Entrena modelo para diferencial de puntos con clusters"""
    features = [col for col in regression_data.columns if col not in parameters["targets"].values()]
    target_col = parameters["targets"]["point_differential"]
    
    X = regression_data[features]
    y = regression_data[target_col].values
    
    results = evaluate_models_with_clusters(X, y, "point_differential", parameters)
    return results["best_model"], results

def create_model_report(local_results: Dict, away_results: Dict, 
                       diff_results: Dict, parameters: Dict) -> Dict:
    """Crea reporte consolidado incluyendo análisis de clusters"""
    logger.info("Creando reporte de modelos con análisis de clusters")
    
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
        "cluster_analysis": {
            "local_strength": local_results.get("cluster_analysis", {}),
            "away_weakness": away_results.get("cluster_analysis", {}),
            "point_differential": diff_results.get("cluster_analysis", {})
        },
        "cluster_features_used": {
            "local_strength": local_results.get("cluster_features_used", []),
            "away_weakness": away_results.get("cluster_features_used", []),
            "point_differential": diff_results.get("cluster_features_used", [])
        },
        "parameters_used": parameters
    }
    
    # Análisis de mejora por clusters
    improvement_analysis = _analyze_cluster_improvement(report)
    report["improvement_analysis"] = improvement_analysis
    
    logger.info("Reporte de modelos con clusters creado exitosamente")
    return report

def _analyze_cluster_improvement(report: Dict) -> Dict:
    """Analiza la mejora aportada por las features de clustering"""
    analysis = {}
    
    for target in ["local_strength", "away_weakness", "point_differential"]:
        cluster_features = report["cluster_features_used"][target]
        cluster_analysis = report["cluster_analysis"][target]
        
        analysis[target] = {
            "n_cluster_features": len(cluster_features),
            "cluster_features": cluster_features,
            "model_importance_analysis": cluster_analysis
        }
    
    return analysis