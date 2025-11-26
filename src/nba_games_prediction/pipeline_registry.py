from kedro.pipeline import Pipeline
from nba_games_prediction.pipelines import (
    data_engineering, 
    classification, 
    regression, 
    clustering,
    dimensionality_reduction,  # NUEVO
    anomaly_detection          # NUEVO
)

def register_pipelines() -> dict:
    """Register the project's pipelines."""
    
    data_engineering_pipeline = data_engineering.create_pipeline()
    classification_pipeline = classification.create_pipeline()
    regression_pipeline = regression.create_pipeline()
    clustering_pipeline = clustering.create_pipeline()
    dimensionality_pipeline = dimensionality_reduction.create_pipeline()  # NUEVO
    anomaly_pipeline = anomaly_detection.create_pipeline()  # NUEVO
    
    # Pipeline completamente integrado con todas las dependencias
    full_integrated_pipeline = (
        data_engineering_pipeline +
        clustering_pipeline +
        dimensionality_pipeline +
        anomaly_pipeline +
        regression_pipeline +
        classification_pipeline
    )
    
    # Pipeline solo con unsupervised learning
    unsupervised_pipeline = (
        data_engineering_pipeline +
        clustering_pipeline +
        dimensionality_pipeline +
        anomaly_pipeline
    )
    
    # Pipeline solo con supervised learning (original)
    supervised_pipeline = (
        data_engineering_pipeline +
        regression_pipeline +
        classification_pipeline
    )
    
    # Pipeline con integración básica (clustering + supervised)
    integrated_basic_pipeline = (
        data_engineering_pipeline +
        clustering_pipeline +
        regression_pipeline +
        classification_pipeline
    )

    return {
        # Pipeline por defecto - TODOS los componentes integrados
        "__default__": full_integrated_pipeline,
        
        # Pipelines individuales
        "de": data_engineering_pipeline,
        "clf": classification_pipeline,
        "reg": regression_pipeline,
        "clustering": clustering_pipeline,
        "dimensionality": dimensionality_pipeline, 
        "anomaly": anomaly_pipeline, 
        
        # Pipelines combinados
        "full": full_integrated_pipeline,
        "unsupervised_only": unsupervised_pipeline,  
        "supervised_only": supervised_pipeline,
        "integrated_basic": integrated_basic_pipeline, 
        
        # Pipelines específicos para testing
        "clustering_dimensionality": clustering_pipeline + dimensionality_pipeline,
        "anomaly_only": data_engineering_pipeline + anomaly_pipeline,
    }