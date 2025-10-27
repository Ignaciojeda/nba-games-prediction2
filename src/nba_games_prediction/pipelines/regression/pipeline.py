"""
This is a boilerplate pipeline 'regression'
generated using Kedro 1.0.0
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    prepare_regression_data,
    train_local_strength_model,
    train_away_weakness_model,
    train_point_differential_model,
    create_model_report
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_regression_data,
                inputs=["games_teams_details", "params:regression"],
                outputs="regression_data",
                name="prepare_regression_data_node",
            ),
            node(
                func=train_local_strength_model,
                inputs=["regression_data", "params:regression"],
                outputs=["local_strength_model", "local_strength_results"],
                name="train_local_strength_model_node",
            ),
            node(
                func=train_away_weakness_model,
                inputs=["regression_data", "params:regression"],
                outputs=["away_weakness_model", "away_weakness_results"],
                name="train_away_weakness_model_node",
            ),
            node(
                func=train_point_differential_model,
                inputs=["regression_data", "params:regression"],
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
                outputs="model_report",
                name="create_model_report_node",
            ),
        ]
    )