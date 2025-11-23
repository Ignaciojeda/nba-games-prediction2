# src/nba_games_prediction/pipelines/clustering/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_clustering_data,
    perform_team_segmentation,
    elbow_analysis,
    silhouette_analysis,
    interpret_clusters,
    evaluate_clustering_models
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the clustering pipeline."""
    return Pipeline(
        [
            # 1. Preparar datos para clustering
            node(
                func=prepare_clustering_data,
                inputs=["team_features_base", "params:clustering"],
                outputs="clustering_data",
                name="prepare_clustering_data_node"
            ),
            
            # 2. Análisis del codo para determinar k óptimo
            node(
                func=elbow_analysis,
                inputs=["clustering_data", "params:clustering"],
                outputs="elbow_analysis_results",
                name="elbow_analysis_node"
            ),
            
            # 3. Análisis de silueta
            node(
                func=silhouette_analysis,
                inputs=["clustering_data", "params:clustering"],
                outputs="silhouette_analysis_results", 
                name="silhouette_analysis_node"
            ),
            
            # 4. Segmentación de equipos con múltiples algoritmos
            node(
                func=perform_team_segmentation,
                inputs=["clustering_data", "params:clustering", "elbow_analysis_results"],
                outputs="clustering_results",
                name="team_segmentation_node"
            ),
            
            # 5. Evaluar modelos de clustering
            node(
                func=evaluate_clustering_models,
                inputs=["clustering_results", "clustering_data", "params:clustering"],
                outputs="clustering_evaluation",
                name="evaluate_clustering_models_node"
            ),
            
            # 6. Interpretar clusters y generar insights
            node(
                func=interpret_clusters,
                inputs=["clustering_results", "team_features_base", "clustering_evaluation", "params:clustering"],
                outputs="cluster_interpretation_report",
                name="interpret_clusters_node"
            ),
        ]
    )