"""
Nodes para detección de anomalías - NBA Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List
import logging

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

def detect_anomalies_isolation_forest(clustering_data: pd.DataFrame, params: Dict) -> Dict:
    """Detección de anomalías usando Isolation Forest."""
    logger.info("Aplicando Isolation Forest para detección de anomalías")
    
    contamination = params.get("contamination", 0.1)
    n_estimators = params.get("n_estimators", 100)
    random_state = params.get("random_state", 42)
    
    # Estandarizar datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(clustering_data)
    
    # Aplicar Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    anomaly_scores = iso_forest.fit_predict(data_scaled)
    decision_scores = iso_forest.decision_function(data_scaled)
    
    # Convertir a etiquetas (1 = normal, -1 = anomalía)
    anomaly_labels = np.where(anomaly_scores == 1, 0, 1)  # 0 = normal, 1 = anomalía
    
    results = {
        'model': iso_forest,
        'anomaly_labels': anomaly_labels.tolist(),
        'anomaly_scores': decision_scores.tolist(),
        'contamination': contamination,
        'n_anomalies': np.sum(anomaly_labels),
        'n_normal': len(anomaly_labels) - np.sum(anomaly_labels),
        'scaler': scaler,
        'algorithm': 'IsolationForest'
    }
    
    logger.info(f"Isolation Forest: {np.sum(anomaly_labels)} anomalías detectadas")
    
    return results

def detect_anomalies_lof(clustering_data: pd.DataFrame, params: Dict) -> Dict:
    """Detección de anomalías usando Local Outlier Factor (LOF)."""
    logger.info("Aplicando Local Outlier Factor para detección de anomalías")
    
    contamination = params.get("contamination", 0.1)
    n_neighbors = params.get("n_neighbors", 20)
    
    # Estandarizar datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(clustering_data)
    
    # Aplicar LOF
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1
    )
    
    anomaly_labels = lof.fit_predict(data_scaled)
    anomaly_scores = lof.negative_outlier_factor_
    
    # Convertir a etiquetas (1 = normal, -1 = anomalía)
    anomaly_labels = np.where(anomaly_labels == 1, 0, 1)  # 0 = normal, 1 = anomalía
    
    results = {
        'model': lof,
        'anomaly_labels': anomaly_labels.tolist(),
        'anomaly_scores': anomaly_scores.tolist(),
        'contamination': contamination,
        'n_neighbors': n_neighbors,
        'n_anomalies': np.sum(anomaly_labels),
        'n_normal': len(anomaly_labels) - np.sum(anomaly_labels),
        'scaler': scaler,
        'algorithm': 'LocalOutlierFactor'
    }
    
    logger.info(f"LOF: {np.sum(anomaly_labels)} anomalías detectadas")
    
    return results

def detect_anomalies_one_class_svm(clustering_data: pd.DataFrame, params: Dict) -> Dict:
    """Detección de anomalías usando One-Class SVM."""
    logger.info("Aplicando One-Class SVM para detección de anomalías")
    
    nu = params.get("contamination", 0.1)  # nu es equivalente a contamination en OCSVM
    kernel = params.get("kernel", "rbf")
    
    # Estandarizar datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(clustering_data)
    
    # Aplicar One-Class SVM (sin random_state)
    oc_svm = OneClassSVM(
        nu=nu,
        kernel=kernel
        # OneClassSVM no tiene parámetro random_state
    )
    
    anomaly_labels = oc_svm.fit_predict(data_scaled)
    decision_scores = oc_svm.decision_function(data_scaled)
    
    # Convertir a etiquetas (1 = normal, -1 = anomalía)
    anomaly_labels = np.where(anomaly_labels == 1, 0, 1)  # 0 = normal, 1 = anomalía
    
    results = {
        'model': oc_svm,
        'anomaly_labels': anomaly_labels.tolist(),
        'anomaly_scores': decision_scores.tolist(),
        'nu': nu,
        'kernel': kernel,
        'n_anomalies': np.sum(anomaly_labels),
        'n_normal': len(anomaly_labels) - np.sum(anomaly_labels),
        'scaler': scaler,
        'algorithm': 'OneClassSVM'
    }
    
    logger.info(f"One-Class SVM: {np.sum(anomaly_labels)} anomalías detectadas")
    
    return results

def analyze_anomalies(iso_forest_results: Dict, lof_results: Dict, 
                     oc_svm_results: Dict, team_features_base: pd.DataFrame) -> Dict:
    """Analizar y comparar resultados de diferentes algoritmos de detección de anomalías."""
    logger.info("Analizando y comparando anomalías detectadas")
    
    # Crear DataFrame con resultados de todos los algoritmos
    comparison_data = []
    
    algorithms = {
        'IsolationForest': iso_forest_results,
        'LocalOutlierFactor': lof_results, 
        'OneClassSVM': oc_svm_results
    }
    
    for algo_name, results in algorithms.items():
        anomaly_labels = np.array(results['anomaly_labels'])
        n_anomalies = np.sum(anomaly_labels)
        
        comparison_data.append({
            'algorithm': algo_name,
            'n_anomalies': int(n_anomalies),
            'n_normal': int(len(anomaly_labels) - n_anomalies),
            'anomaly_percentage': float(n_anomalies / len(anomaly_labels) * 100),
            'contamination': results.get('contamination', results.get('nu', 0.1))
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Identificar equipos anómalos comunes entre algoritmos
    iso_labels = np.array(iso_forest_results['anomaly_labels'])
    lof_labels = np.array(lof_results['anomaly_labels']) 
    svm_labels = np.array(oc_svm_results['anomaly_labels'])
    
    # Equipos detectados como anomalías por múltiples algoritmos
    consensus_anomalies = (iso_labels + lof_labels + svm_labels) >= 2
    n_consensus = np.sum(consensus_anomalies)
    
    # Obtener nombres de equipos anómalos
    anomaly_indices = np.where(consensus_anomalies)[0]
    if len(anomaly_indices) > 0 and 'TEAM_ID' in team_features_base.columns:
        anomaly_teams = team_features_base.iloc[anomaly_indices]['TEAM_ID'].tolist()
    else:
        anomaly_teams = []
    
    analysis_report = {
        'algorithm_comparison': comparison_df.to_dict('records'),
        'consensus_anomalies': {
            'n_anomalies': int(n_consensus),
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_teams': anomaly_teams,
            'consensus_threshold': 2
        },
        'individual_results': {
            'IsolationForest': {
                'n_anomalies': iso_forest_results['n_anomalies'],
                'anomaly_indices': np.where(iso_labels == 1)[0].tolist()
            },
            'LocalOutlierFactor': {
                'n_anomalies': lof_results['n_anomalies'],
                'anomaly_indices': np.where(lof_labels == 1)[0].tolist()
            },
            'OneClassSVM': {
                'n_anomalies': oc_svm_results['n_anomalies'],
                'anomaly_indices': np.where(svm_labels == 1)[0].tolist()
            }
        }
    }
    
    logger.info(f"Análisis completado: {n_consensus} anomalías por consenso")
    
    return analysis_report

def integrate_anomaly_scores(iso_forest_results: Dict, team_features_base: pd.DataFrame) -> pd.DataFrame:
    """Integrar scores de anomalías como features para modelos supervisados."""
    logger.info("Integrando scores de anomalías como features")
    
    # Usar Isolation Forest como referencia principal
    anomaly_scores = iso_forest_results['anomaly_scores']
    anomaly_labels = iso_forest_results['anomaly_labels']
    
    # Crear features mejoradas
    enhanced_features = team_features_base.copy()
    
    # Agregar scores y labels de anomalías
    enhanced_features['ANOMALY_SCORE'] = anomaly_scores
    enhanced_features['IS_ANOMALY'] = anomaly_labels
    enhanced_features['ANOMALY_CONFIDENCE'] = 1 - (1 / (1 + np.exp(-np.abs(anomaly_scores))))
    
    logger.info(f"Features mejoradas con scores de anomalías: {enhanced_features.shape}")
    logger.info(f"Equipos anómalos detectados: {np.sum(anomaly_labels)}")
    
    return enhanced_features