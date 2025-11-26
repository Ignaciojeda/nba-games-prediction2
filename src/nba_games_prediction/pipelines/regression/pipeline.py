"""
Pipeline de regresión actualizado con integración de clustering
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    prepare_regression_data_with_clusters,
    train_local_strength_model,
    train_away_weakness_model,
    train_point_differential_model,
    create_model_report
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_regression_data_with_clusters,
                inputs=[
                    "games_teams_details", 
                    "clustering_results",
                    "team_features_base",  # NUEVO INPUT
                    "params:regression"
                ],
                outputs="regression_data_with_clusters",
                name="prepare_regression_data_with_clusters_node",
            ),
            node(
                func=train_local_strength_model,
                inputs=["regression_data_with_clusters", "params:regression"],
                outputs=["local_strength_model", "local_strength_results"],
                name="train_local_strength_model_node",
            ),
            node(
                func=train_away_weakness_model,
                inputs=["regression_data_with_clusters", "params:regression"],
                outputs=["away_weakness_model", "away_weakness_results"],
                name="train_away_weakness_model_node",
            ),
            node(
                func=train_point_differential_model,
                inputs=["regression_data_with_clusters", "params:regression"],
                outputs=["point_differential_model", "point_differential_results"],
                name="train_point_differential_model_node",
            ),
            node(
                func=create_model_report,
                inputs=[
                    "local_strength_results", 
                    "away_weakness_results", 
                    "point_differential_results",
                    "params:regression"
                ],
                outputs="model_report_with_clusters",
                name="create_model_report_node",
            ),
        ]
    )