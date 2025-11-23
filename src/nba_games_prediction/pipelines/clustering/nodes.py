"""
Nodes for clustering pipeline - NBA team segmentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List
import logging

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def prepare_clustering_data(team_features_base: pd.DataFrame, clustering_params: Dict) -> pd.DataFrame:
    """
    Preparar datos específicos para análisis de clustering.
    
    Args:
        team_features_base: DataFrame con características base de equipos
        clustering_params: Parámetros de configuración para clustering
        
    Returns:
        DataFrame preparado para clustering
    """
    logger.info("Preparando datos para clustering de equipos NBA")
    
    # Seleccionar features para clustering
    clustering_features = clustering_params.get("features", [])
    
    # Verificar que las features existan en el DataFrame
    available_features = [f for f in clustering_features if f in team_features_base.columns]
    
    if not available_features:
        logger.warning("No se encontraron features especificadas. Usando todas las numéricas.")
        available_features = team_features_base.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Usando {len(available_features)} features para clustering: {available_features}")
    
    # Crear dataset para clustering
    clustering_data = team_features_base[available_features].copy()
    
    # Manejar valores nulos
    clustering_data = clustering_data.fillna(clustering_data.median())
    
    # Estandarizar datos
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)
    clustering_data = pd.DataFrame(clustering_data_scaled, 
                                 columns=available_features,
                                 index=team_features_base.index)
    
    logger.info(f"Datos preparados: {clustering_data.shape}")
    logger.info(f"Equipos únicos: {len(clustering_data)}")
    
    return clustering_data


def elbow_analysis(clustering_data: pd.DataFrame, clustering_params: Dict) -> Dict:
    """
    Realizar análisis del codo para determinar número óptimo de clusters.
    
    Args:
        clustering_data: Datos preparados para clustering
        clustering_params: Parámetros de configuración
        
    Returns:
        Dict con resultados del análisis del codo
    """
    logger.info("Realizando análisis del codo")
    
    max_k = clustering_params.get("max_k", 10)
    random_state = clustering_params.get("random_state", 42)
    
    wcss = []  # Within-Cluster Sum of Square
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(clustering_data)
        wcss.append(kmeans.inertia_)
    
    # Calcular la "curvatura" para sugerir k óptimo
    differences = []
    for i in range(1, len(wcss)):
        differences.append(wcss[i-1] - wcss[i])
    
    # Encontrar el punto de mayor curvatura (método alternativo)
    curvatures = []
    for i in range(1, len(differences)-1):
        curvatures.append(differences[i-1] - differences[i])
    
    suggested_k = np.argmax(curvatures) + 2 if curvatures else 3
    
    elbow_results = {
        'k_range': list(k_range),
        'wcss': wcss,
        'differences': differences,
        'curvatures': curvatures,
        'suggested_k': suggested_k,
        'max_k': max_k
    }
    
    logger.info(f"Análisis del codo completado. k sugerido: {suggested_k}")
    
    return elbow_results


def silhouette_analysis(clustering_data: pd.DataFrame, clustering_params: Dict) -> Dict:
    """
    Realizar análisis de silueta para validar calidad de clusters.
    
    Args:
        clustering_data: Datos preparados para clustering
        clustering_params: Parámetros de configuración
        
    Returns:
        Dict con resultados del análisis de silueta
    """
    logger.info("Realizando análisis de silueta")
    
    max_k = clustering_params.get("max_k", 10)
    random_state = clustering_params.get("random_state", 42)
    
    silhouette_scores = []
    k_range = range(2, max_k + 1)  # Silhouette requiere al menos 2 clusters
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(clustering_data)
        silhouette_avg = silhouette_score(clustering_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Encontrar k con mejor score de silueta
    best_k_silhouette = k_range[np.argmax(silhouette_scores)]
    best_silhouette_score = max(silhouette_scores)
    
    silhouette_results = {
        'k_range': list(k_range),
        'silhouette_scores': silhouette_scores,
        'best_k': best_k_silhouette,
        'best_score': best_silhouette_score
    }
    
    logger.info(f"Análisis de silueta completado. Mejor k: {best_k_silhouette} (score: {best_silhouette_score:.4f})")
    
    return silhouette_results


def perform_team_segmentation(clustering_data: pd.DataFrame, 
                            clustering_params: Dict,
                            elbow_results: Dict) -> Dict:
    """
    Realizar segmentación de equipos usando múltiples algoritmos de clustering.
    
    Args:
        clustering_data: Datos preparados para clustering
        clustering_params: Parámetros de configuración
        elbow_results: Resultados del análisis del codo
        
    Returns:
        Dict con resultados de todos los algoritmos de clustering
    """
    logger.info("Realizando segmentación de equipos con múltiples algoritmos")
    
    random_state = clustering_params.get("random_state", 42)
    n_clusters = clustering_params.get("n_clusters") or elbow_results.get('suggested_k', 3)
    
    results = {}
    
    # 1. K-Means Clustering
    logger.info(f"Aplicando K-Means con k={n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(clustering_data)
    
    results['kmeans'] = {
        'model': kmeans,
        'labels': kmeans_labels,
        'centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'n_clusters': n_clusters
    }
    
    # 2. DBSCAN Clustering
    logger.info("Aplicando DBSCAN")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(clustering_data)
    
    # Ajustar número de clusters encontrados por DBSCAN
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    
    results['dbscan'] = {
        'model': dbscan,
        'labels': dbscan_labels,
        'n_clusters': n_clusters_dbscan,
        'noise_points': list(dbscan_labels).count(-1)
    }
    
    # 3. Agglomerative Clustering
    logger.info(f"Aplicando Agglomerative Clustering con k={n_clusters}")
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    agglo_labels = agglo.fit_predict(clustering_data)
    
    results['agglo'] = {
        'model': agglo,
        'labels': agglo_labels,
        'n_clusters': n_clusters
    }
    
    # 4. PCA + K-Means (reducción dimensionalidad)
    logger.info("Aplicando PCA + K-Means")
    pca = PCA(n_components=0.95, random_state=random_state)  # 95% varianza
    clustering_data_pca = pca.fit_transform(clustering_data)
    
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_pca_labels = kmeans_pca.fit_predict(clustering_data_pca)
    
    results['pca_kmeans'] = {
        'model': kmeans_pca,
        'labels': kmeans_pca_labels,
        'pca_model': pca,
        'n_components': pca.n_components_,
        'explained_variance': pca.explained_variance_ratio_.sum()
    }
    
    logger.info("Segmentación completada con todos los algoritmos")
    
    return results


def evaluate_clustering_models(clustering_results: Dict, 
                             clustering_data: pd.DataFrame,
                             clustering_params: Dict) -> Dict:
    """
    Evaluar y comparar los diferentes modelos de clustering.
    
    Args:
        clustering_results: Resultados de todos los algoritmos
        clustering_data: Datos utilizados para clustering
        clustering_params: Parámetros de configuración
        
    Returns:
        Dict con evaluación comparativa de modelos
    """
    logger.info("Evaluando modelos de clustering")
    
    evaluation = {}
    
    for algo_name, algo_results in clustering_results.items():
        labels = algo_results['labels']
        
        # Solo evaluar si hay al menos 2 clusters y no todos son noise
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters >= 2:
            # Calcular métricas de evaluación
            silhouette_avg = silhouette_score(clustering_data, labels)
            calinski_harabasz = calinski_harabasz_score(clustering_data, labels)
            davies_bouldin = davies_bouldin_score(clustering_data, labels)
            
            evaluation[algo_name] = {
                'silhouette_score': silhouette_avg,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'n_clusters': n_clusters,
                'n_noise': list(labels).count(-1) if -1 in labels else 0
            }
            
            logger.info(f"{algo_name}: Silhouette={silhouette_avg:.4f}, Clusters={n_clusters}")
        else:
            evaluation[algo_name] = {
                'error': 'No enough clusters for evaluation',
                'n_clusters': n_clusters
            }
            logger.warning(f"{algo_name}: No se pudo evaluar - muy pocos clusters")
    
    # Determinar mejor algoritmo basado en silhouette score
    valid_algos = {k: v for k, v in evaluation.items() 
                  if 'silhouette_score' in v and v['n_clusters'] >= 2}
    
    if valid_algos:
        best_algo = max(valid_algos.keys(), 
                       key=lambda x: valid_algos[x]['silhouette_score'])
        evaluation['best_algorithm'] = best_algo
        evaluation['best_score'] = valid_algos[best_algo]['silhouette_score']
        
        logger.info(f"Mejor algoritmo: {best_algo} (Silhouette: {evaluation['best_score']:.4f})")
    else:
        evaluation['best_algorithm'] = 'kmeans'  # Default
        evaluation['best_score'] = 0.0
        logger.warning("No se pudo determinar el mejor algoritmo")
    
    return evaluation


def interpret_clusters(clustering_results: Dict,
                      team_features_base: pd.DataFrame,
                      clustering_evaluation: Dict,
                      clustering_params: Dict) -> Dict:
    """
    Interpretar los clusters encontrados y generar insights de negocio.
    
    Args:
        clustering_results: Resultados de clustering
        team_features_base: Datos originales de equipos
        clustering_evaluation: Evaluación de modelos
        clustering_params: Parámetros de configuración
        
    Returns:
        Dict con interpretación y insights de clusters
    """
    logger.info("Interpretando clusters y generando insights")
    
    # Usar el mejor algoritmo según evaluación
    best_algo = clustering_evaluation.get('best_algorithm', 'kmeans')
    best_results = clustering_results[best_algo]
    cluster_labels = best_results['labels']
    
    # Agregar labels al dataset original
    team_data_with_clusters = team_features_base.copy()
    team_data_with_clusters['cluster'] = cluster_labels
    
    # Analizar cada cluster
    cluster_analysis = {}
    
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:  # Skip noise points
            continue
            
        cluster_data = team_data_with_clusters[team_data_with_clusters['cluster'] == cluster_id]
        
        # Estadísticas descriptivas del cluster
        cluster_stats = cluster_data.describe().to_dict()
        
        # Características más distintivas
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
        if 'cluster' in numeric_cols:
            numeric_cols = numeric_cols.drop('cluster')
        
        overall_means = team_features_base[numeric_cols].mean()
        cluster_means = cluster_data[numeric_cols].mean()
        
        # Encontrar características que más difieren
        differences = (cluster_means - overall_means).abs()
        top_features = differences.nlargest(5).index.tolist()
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_data),
            'teams': cluster_data.index.tolist() if hasattr(cluster_data.index, 'tolist') else list(cluster_data.index),
            'statistics': cluster_stats,
            'distinctive_features': top_features,
            'feature_differences': differences.to_dict(),
            'description': _generate_cluster_description(cluster_means, overall_means, top_features)
        }
    
    # Generar reporte consolidado
    interpretation_report = {
        'best_algorithm': best_algo,
        'n_clusters': len([x for x in set(cluster_labels) if x != -1]),
        'cluster_analysis': cluster_analysis,
        'algorithm_performance': clustering_evaluation,
        'business_insights': _generate_business_insights(cluster_analysis),
        'recommendations': _generate_strategic_recommendations(cluster_analysis)
    }
    
    logger.info("Interpretación de clusters completada")
    
    return interpretation_report


def _generate_cluster_description(cluster_means: pd.Series, 
                                overall_means: pd.Series,
                                top_features: List[str]) -> str:
    """Generar descripción automática del cluster basado en características."""
    
    descriptions = []
    
    for feature in top_features[:3]:  # Top 3 características
        cluster_val = cluster_means[feature]
        overall_val = overall_means[feature]
        diff_pct = ((cluster_val - overall_val) / overall_val) * 100
        
        if diff_pct > 20:
            descriptions.append(f"muy alto en {feature}")
        elif diff_pct > 10:
            descriptions.append(f"alto en {feature}")
        elif diff_pct < -20:
            descriptions.append(f"muy bajo en {feature}")
        elif diff_pct < -10:
            descriptions.append(f"bajo en {feature}")
    
    if descriptions:
        return f"Equipos con {', '.join(descriptions)}"
    else:
        return "Equipos con características promedio"


def _generate_business_insights(cluster_analysis: Dict) -> List[str]:
    """Generar insights de negocio basados en los clusters."""
    
    insights = []
    
    for cluster_id, analysis in cluster_analysis.items():
        size = analysis['size']
        desc = analysis['description']
        
        insights.append(
            f"Cluster {cluster_id} ({size} equipos): {desc}. "
            f"Estos equipos podrían beneficiarse de estrategias específicas."
        )
    
    # Insight comparativo
    if len(cluster_analysis) > 1:
        largest_cluster = max(cluster_analysis.items(), key=lambda x: x[1]['size'])
        smallest_cluster = min(cluster_analysis.items(), key=lambda x: x[1]['size'])
        
        insights.append(
            f"El cluster {largest_cluster[0]} es el más común ({largest_cluster[1]['size']} equipos), "
            f"mientras que el cluster {smallest_cluster[0]} es el más exclusivo ({smallest_cluster[1]['size']} equipos)."
        )
    
    return insights


def _generate_strategic_recommendations(cluster_analysis: Dict) -> List[str]:
    """Generar recomendaciones estratégicas basadas en los clusters."""
    
    recommendations = []
    
    for cluster_id, analysis in cluster_analysis.items():
        distinctive_features = analysis['distinctive_features']
        
        if 'offensive' in str(distinctive_features).lower() or 'points' in str(distinctive_features).lower():
            recommendations.append(
                f"Cluster {cluster_id}: Enfocar en mejorar la defensa y mantener la eficiencia ofensiva."
            )
        elif 'defensive' in str(distinctive_features).lower():
            recommendations.append(
                f"Cluster {cluster_id}: Desarrollar más variantes ofensivas sin comprometer la defensa sólida."
            )
        else:
            recommendations.append(
                f"Cluster {cluster_id}: Buscar ventajas competitivas en áreas específicas basadas en su perfil único."
            )
    
    recommendations.append(
        "Considerar scouting de jugadores que se alineen con el perfil deseado de cada cluster."
    )
    recommendations.append(
        "Desarrollar estrategias de juego específicas para enfrentar equipos de diferentes clusters."
    )
    
    return recommendations