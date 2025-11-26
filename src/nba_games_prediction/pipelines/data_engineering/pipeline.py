"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    clean_games_data,
    clean_teams_data,
    clean_games_details_data,
    handle_missing_values,
    create_eda_visualizations,
    create_team_features_base,
    create_game_level_features,
    prepare_model_inputs,
    validate_data_quality,
    create_games_teams_details
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_games_data,
                inputs="raw_games",
                outputs="games_cleaned",
                name="clean_games_data_node",
            ),
            node(
                func=clean_teams_data,
                inputs="raw_teams", 
                outputs="teams_cleaned",
                name="clean_teams_data_node",
            ),
            node(
                func=clean_games_details_data,
                inputs="raw_games_details",
                outputs="games_details_cleaned", 
                name="clean_games_details_node",
            ),
            node(
                func=handle_missing_values,
                inputs="games_cleaned",
                outputs="games_validated",
                name="handle_missing_values_node",
            ),
            node(
                func=create_eda_visualizations,
                inputs="games_validated",
                outputs="eda_visualizations",
                name="create_eda_visualizations_node",
            ),
            node(
                func=create_games_teams_details,
                inputs=["games_validated", "teams_cleaned", "games_details_cleaned"],
                outputs="games_teams_details", 
                name="create_games_teams_details_node",
            ),
            node(
                func=create_team_features_base,
                inputs=["games_validated", "teams_cleaned"],
                outputs="team_features_base",
                name="create_team_features_base_node",
            ),
            node(
                func=create_game_level_features,
                inputs=["games_validated", "team_features_base"],
                outputs="game_level_features",
                name="create_game_level_features_node", 
            ),
            node(
                func=prepare_model_inputs,
                inputs=["game_level_features", "params:data_engineering"],
                outputs=["model_input_classification", "model_input_regression"],
                name="prepare_model_inputs_node",
            ),
            node(
                func=validate_data_quality,
                inputs=["games_validated", "team_features_base", "game_level_features"],
                outputs="validation_report",
                name="validate_data_quality_node",
            ),
        ]
    )