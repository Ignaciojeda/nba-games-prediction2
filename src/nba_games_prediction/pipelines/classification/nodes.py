"""
Nodes para pipeline de clasificación - Predicción de victorias NBA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import logging
import json
from typing import Dict, Tuple, Any, List
import warnings
import pickle
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def prepare_classification_data_with_clusters(
    classification_data: pd.DataFrame,
    clustering_results: Dict,
    team_features_base: pd.DataFrame
) -> Tuple:
    """
    Preparar datos para clasificación integrando información de clusters
    """
    logger.info("Preparando datos para clasificación con clusters")
    
    # Verificar que tenemos las columnas necesarias
    required_columns = ['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS']
    missing_columns = [col for col in required_columns if col not in classification_data.columns]
    
    if missing_columns:
        logger.error(f"Columnas requeridas faltantes: {missing_columns}")
        raise ValueError(f"Faltan columnas esenciales: {missing_columns}")
    
    # Separar features y target
    feature_columns = [col for col in classification_data.columns if col not in ['HOME_TEAM_WINS']]
    X = classification_data[feature_columns]
    y = classification_data['HOME_TEAM_WINS']
    
    # Mejorar con features de clustering
    X_enhanced = _enhance_classification_with_clusters(X, clustering_results, team_features_base)
    
    logger.info(f"Features originales: {X.shape[1]}, Con clusters: {X_enhanced.shape[1]}")
    logger.info(f"Distribución del target: {y.value_counts().to_dict()}")
    logger.info(f"Nuevas features agregadas: {[col for col in X_enhanced.columns if col not in X.columns]}")
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("✅ Datos preparados para clasificación con clusters")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_enhanced.columns.tolist()

def _enhance_classification_with_clusters(
    X: pd.DataFrame, 
    clustering_results: Dict,
    team_features_base: pd.DataFrame
) -> pd.DataFrame:
    """
    Mejorar features de clasificación con información de clusters
    """
    X_enhanced = X.copy()
    all_columns = X_enhanced.columns.tolist()
    
    # Obtener el mejor algoritmo de clustering
    best_algo = clustering_results.get('best_algorithm', 'kmeans')
    best_results = clustering_results[best_algo]
    cluster_labels = best_results['labels']
    
    # Crear mapeo de TEAM_ID a cluster
    team_cluster_map = {}
    if 'TEAM_ID' in team_features_base.columns:
        # Asumimos que team_features_base está en el mismo orden que los clusters
        for i, team_id in enumerate(team_features_base['TEAM_ID'].values):
            if i < len(cluster_labels):
                team_cluster_map[team_id] = cluster_labels[i]
    
    logger.info(f"Mapeo de equipos a clusters creado: {len(team_cluster_map)} equipos")
    
    # Agregar clusters como features usando HOME_TEAM_ID y VISITOR_TEAM_ID
    if team_cluster_map and 'HOME_TEAM_ID' in X_enhanced.columns and 'VISITOR_TEAM_ID' in X_enhanced.columns:
        
        # Agregar cluster del equipo local
        X_enhanced['HOME_TEAM_CLUSTER'] = X_enhanced['HOME_TEAM_ID'].map(team_cluster_map)
        # Agregar cluster del equipo visitante
        X_enhanced['AWAY_TEAM_CLUSTER'] = X_enhanced['VISITOR_TEAM_ID'].map(team_cluster_map)
        
        # Llenar NaN con -1 (para equipos no encontrados)
        X_enhanced['HOME_TEAM_CLUSTER'] = X_enhanced['HOME_TEAM_CLUSTER'].fillna(-1)
        X_enhanced['AWAY_TEAM_CLUSTER'] = X_enhanced['AWAY_TEAM_CLUSTER'].fillna(-1)
        
        logger.info(f"Clusters asignados - Local: {X_enhanced['HOME_TEAM_CLUSTER'].value_counts().to_dict()}")
        logger.info(f"Clusters asignados - Visitante: {X_enhanced['AWAY_TEAM_CLUSTER'].value_counts().to_dict()}")
        
        # Crear variables dummy para clusters
        home_cluster_dummies = pd.get_dummies(X_enhanced['HOME_TEAM_CLUSTER'], prefix='HOME_CLUSTER')
        away_cluster_dummies = pd.get_dummies(X_enhanced['AWAY_TEAM_CLUSTER'], prefix='AWAY_CLUSTER')
        
        X_enhanced = pd.concat([X_enhanced, home_cluster_dummies, away_cluster_dummies], axis=1)
        
        # Agregar feature de si están en el mismo cluster
        X_enhanced['SAME_CLUSTER'] = (X_enhanced['HOME_TEAM_CLUSTER'] == X_enhanced['AWAY_TEAM_CLUSTER']).astype(int)
        
        # Agregar diferencia de clusters
        X_enhanced['CLUSTER_DIFF'] = X_enhanced['HOME_TEAM_CLUSTER'] - X_enhanced['AWAY_TEAM_CLUSTER']
        
        # Agregar características de cluster (estadísticas)
        cluster_stats = _calculate_cluster_statistics_for_classification(clustering_results, team_features_base)
        if not cluster_stats.empty:
            # Unir estadísticas para equipo local
            X_enhanced = X_enhanced.merge(
                cluster_stats.add_prefix('HOME_CLUSTER_STATS_'), 
                left_on='HOME_TEAM_CLUSTER', 
                right_index=True, 
                how='left'
            )
            # Unir estadísticas para equipo visitante
            X_enhanced = X_enhanced.merge(
                cluster_stats.add_prefix('AWAY_CLUSTER_STATS_'), 
                left_on='AWAY_TEAM_CLUSTER', 
                right_index=True, 
                how='left'
            )
    else:
        logger.warning("No se pudieron agregar features de clustering - columnas de equipo faltantes")
    
    # Llenar valores NaN que puedan haberse generado
    X_enhanced = X_enhanced.fillna(0)
    
    # Eliminar columnas temporales si existen
    columns_to_drop = ['HOME_TEAM_CLUSTER', 'AWAY_TEAM_CLUSTER']
    X_enhanced = X_enhanced.drop(columns=[col for col in columns_to_drop if col in X_enhanced.columns])
    
    # DEBUG: Mostrar nuevas features agregadas
    new_features = [col for col in X_enhanced.columns if col not in all_columns]
    logger.info(f"✅ Nuevas features agregadas: {len(new_features)}")
    if new_features:
        logger.info(f"Ejemplos: {new_features[:10]}")
    
    return X_enhanced

def _calculate_cluster_statistics_for_classification(clustering_results: Dict, team_features_base: pd.DataFrame) -> pd.DataFrame:
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

def _instantiate_model(model_config: Dict) -> Any:
    """Instanciar modelo desde configuración"""
    model_class_mapping = {
        'sklearn.linear_model.LogisticRegression': LogisticRegression,
        'sklearn.tree.DecisionTreeClassifier': DecisionTreeClassifier,
        'sklearn.ensemble.RandomForestClassifier': RandomForestClassifier,
        'sklearn.ensemble.GradientBoostingClassifier': GradientBoostingClassifier,
        'sklearn.svm.SVC': SVC,
        'sklearn.naive_bayes.GaussianNB': GaussianNB,
        'sklearn.neighbors.KNeighborsClassifier': KNeighborsClassifier
    }
    
    model_class_str = model_config['class']
    if model_class_str not in model_class_mapping:
        raise ValueError(f"Clase de modelo no reconocida: {model_class_str}")
    
    model_class = model_class_mapping[model_class_str]
    model_kwargs = model_config.get('kwargs', {})
    
    return model_class(**model_kwargs)

def train_classification_models(X_train: np.ndarray, y_train: pd.Series, 
                              models_config: Dict) -> Dict[str, Any]:
    """
    Entrenar modelos de clasificación con GridSearchCV
    """
    logger.info("Iniciando entrenamiento de modelos de clasificación con clusters")
    
    trained_models = {}
    
    for model_name, config in models_config.items():
        logger.info(f"Entrenando {model_name}...")
        
        try:
            # Instanciar modelo
            model = _instantiate_model(config['model'])
            
            if config.get('params', {}):  # Si tiene parámetros para GridSearch
                grid_search = GridSearchCV(
                    model, 
                    config['params'], 
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_score = grid_search.best_score_
                
            else:  # Sin GridSearch
                model.fit(X_train, y_train)
                best_model = model
                best_params = "No GridSearch"
                cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))
            
            # Cross-validation scores detallados
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
            
            trained_models[model_name] = {
                'model': best_model,
                'best_params': best_params,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_best': cv_score
            }
            
            logger.info(f"✅ {model_name} - CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
        except Exception as e:
            logger.error(f"Error entrenando {model_name}: {str(e)}")
            continue
    
    return trained_models

def evaluate_classification_models(trained_models: Dict, X_test: np.ndarray, 
                                 y_test: pd.Series, scaler: StandardScaler) -> Dict:
    """
    Evaluar modelos en conjunto de test
    """
    logger.info("Evaluando modelos en conjunto de test con clusters")
    
    evaluation_results = {}
    
    for model_name, model_info in trained_models.items():
        model = model_info['model']
        
        try:
            # Predecir
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc_roc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Reporte de clasificación
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'classification_report': class_report,
                'confusion_matrix': cm.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None,
                'cv_mean': model_info['cv_mean'],
                'cv_std': model_info['cv_std']
            }
            
            # Log de resultados
            auc_display = f"{auc_roc:.4f}" if auc_roc is not None else "N/A"
            logger.info(f"📊 {model_name} - Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_display}")
            
        except Exception as e:
            logger.error(f"Error evaluando {model_name}: {str(e)}")
            continue
    
    return evaluation_results

def analyze_best_model(evaluation_results: Dict, X_test: np.ndarray, 
                      y_test: pd.Series, scaler: StandardScaler) -> Dict:
    """
    Analizar el mejor modelo y generar insights
    """
    logger.info("Analizando mejor modelo con clusters")
    
    if not evaluation_results:
        logger.error("No hay resultados de evaluación para analizar")
        return {}
    
    # Encontrar mejor modelo por accuracy
    best_model_name = max(evaluation_results.keys(), 
                         key=lambda x: evaluation_results[x]['accuracy'])
    
    best_model_metrics = evaluation_results[best_model_name]
    baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())  # Mejor modelo base
    
    analysis = {
        'best_model_name': best_model_name,
        'best_accuracy': best_model_metrics['accuracy'],
        'best_precision': best_model_metrics['precision'],
        'best_recall': best_model_metrics['recall'],
        'best_f1_score': best_model_metrics['f1_score'],
        'best_auc_roc': best_model_metrics['auc_roc'],
        'cv_accuracy': best_model_metrics['cv_mean'],
        'cv_std': best_model_metrics['cv_std'],
        'baseline_accuracy': baseline_accuracy,
        'improvement_absolute': best_model_metrics['accuracy'] - baseline_accuracy,
        'improvement_relative': (best_model_metrics['accuracy'] - baseline_accuracy) / baseline_accuracy * 100,
        'test_set_size': len(y_test),
        'positive_class_ratio': y_test.mean()
    }
    
    logger.info(f"🏆 Mejor modelo: {best_model_name}")
    logger.info(f"📈 Accuracy: {best_model_metrics['accuracy']:.4f} (Baseline: {baseline_accuracy:.4f})")
    logger.info(f"📊 Mejora vs baseline: {analysis['improvement_relative']:.1f}%")
    logger.info(f"🎯 F1-Score: {best_model_metrics['f1_score']:.4f}")
    
    return analysis

def save_classification_results(trained_models: Dict, evaluation_results: Dict,
                              best_model_analysis: Dict, scaler: StandardScaler,
                              models_config: Dict) -> Tuple:
    """
    Guardar resultados, modelos y generar reporte
    """
    logger.info("Guardando resultados de clasificación con clusters")
    
    if not evaluation_results or not best_model_analysis:
        logger.error("No hay resultados para guardar")
        return {}, None, scaler, {}
    
    best_model_name = best_model_analysis['best_model_name']
    best_model = trained_models[best_model_name]['model']
    
    # Feature importance si está disponible
    feature_importance = {}
    if hasattr(best_model, 'feature_importances_'):
        importance_scores = best_model.feature_importances_
        feature_importance = {
            'feature_names': list(range(len(importance_scores))),
            'importance_scores': importance_scores.tolist(),
            'has_importance': True,
            'top_features': np.argsort(importance_scores)[-10:][::-1].tolist()  # Top 10 features
        }
        logger.info(f"🔍 Feature importance disponible para {best_model_name}")
    elif hasattr(best_model, 'coef_'):
        coef = best_model.coef_
        if len(coef.shape) > 1:
            coef = coef[0]  # Para multi-class, tomar primera clase
        feature_importance = {
            'feature_names': list(range(len(coef))),
            'importance_scores': np.abs(coef).tolist(),
            'has_importance': True,
            'top_features': np.argsort(np.abs(coef))[-10:][::-1].tolist()
        }
        logger.info(f"🔍 Coefficients disponibles para {best_model_name}")
    else:
        feature_importance = {
            'has_importance': False,
            'message': f"Modelo {best_model_name} no tiene feature_importances_ ni coef_"
        }
        logger.info(f"⚠️ Feature importance no disponible para {best_model_name}")
    
    # Crear reporte consolidado
    classification_report = {
        'models_trained': list(trained_models.keys()),
        'models_evaluated': list(evaluation_results.keys()),
        'evaluation_results': evaluation_results,
        'best_model_analysis': best_model_analysis,
        'training_config': models_config,
        'summary': {
            'best_model': best_model_name,
            'best_accuracy': best_model_analysis['best_accuracy'],
            'best_f1_score': best_model_analysis['best_f1_score'],
            'best_auc_roc': best_model_analysis['best_auc_roc'],
            'baseline_accuracy': best_model_analysis['baseline_accuracy'],
            'improvement_absolute': best_model_analysis['improvement_absolute'],
            'improvement_relative': best_model_analysis['improvement_relative'],
            'test_set_size': best_model_analysis['test_set_size'],
            'positive_class_ratio': best_model_analysis['positive_class_ratio']
        },
        'feature_analysis': feature_importance,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    logger.info("✅ Resultados de clasificación con clusters guardados exitosamente")
    
    return classification_report, best_model, scaler, feature_importance

def compare_with_original_performance(enhanced_results: Dict, original_results: Dict = None) -> Dict:
    """
    Comparar rendimiento con/sin clusters (si hay resultados originales)
    """
    comparison = {
        'enhanced_performance': enhanced_results['best_model_analysis'],
        'has_comparison': original_results is not None
    }
    
    if original_results:
        original_analysis = original_results['best_model_analysis']
        enhanced_analysis = enhanced_results['best_model_analysis']
        
        comparison['accuracy_comparison'] = {
            'original': original_analysis['best_accuracy'],
            'enhanced': enhanced_analysis['best_accuracy'],
            'improvement': enhanced_analysis['best_accuracy'] - original_analysis['best_accuracy'],
            'improvement_percentage': (
                (enhanced_analysis['best_accuracy'] - original_analysis['best_accuracy']) / 
                original_analysis['best_accuracy'] * 100
            )
        }
        
        comparison['conclusion'] = (
            "Mejora significativa" if comparison['accuracy_comparison']['improvement'] > 0.01 
            else "Mejora marginal" if comparison['accuracy_comparison']['improvement'] > 0 
            else "Sin mejora"
        )
    
    return comparison