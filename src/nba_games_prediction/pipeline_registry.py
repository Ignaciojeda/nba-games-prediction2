from kedro.pipeline import Pipeline
from nba_games_prediction.pipelines import data_engineering, classification, regression

def register_pipelines() -> dict:
    """Register the project's pipelines."""
    
    data_engineering_pipeline = data_engineering.create_pipeline()
    classification_pipeline = classification.create_pipeline()
    regression_pipeline = regression.create_pipeline()
    
    return {
        "__default__": data_engineering_pipeline + classification_pipeline + regression_pipeline,
        "de": data_engineering_pipeline,
        "clf": classification_pipeline,
        "reg": regression_pipeline,
        "full": data_engineering_pipeline + classification_pipeline + regression_pipeline,
    }