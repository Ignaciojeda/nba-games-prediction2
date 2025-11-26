from kedro.pipeline import Pipeline, node, pipeline
from nba_games_prediction.pipelines import (
    data_engineering, 
    classification, 
    regression, 
    clustering,
    dimensionality_reduction,
    anomaly_detection
)

# Importaciones para los nodos específicos
from nba_games_prediction.pipelines.regression.nodes import (
    prepare_regression_data_with_clusters,
    train_local_strength_model,
    train_away_weakness_model, 
    train_point_differential_model,
    create_model_report
)

from nba_games_prediction.pipelines.classification.nodes import (
    prepare_classification_data_with_clusters,
    train_classification_models,
    evaluate_classification_models,
    analyze_best_model,
    save_classification_results
)

def register_pipelines() -> dict:
    """Register the project's pipelines."""
    
    # Pipelines base
    data_engineering_pipeline = data_engineering.create_pipeline()
    classification_pipeline = classification.create_pipeline()
    regression_pipeline = regression.create_pipeline()
    clustering_pipeline = clustering.create_pipeline()
    dimensionality_pipeline = dimensionality_reduction.create_pipeline()
    anomaly_pipeline = anomaly_detection.create_pipeline()
    
    # ========== PIPELINES CON INTEGRACIÓN DE CLUSTERING ==========
    
    # Pipeline de regresión con clusters
    regression_with_clusters_pipeline = Pipeline([
        # Preparar datos de regresión con clusters
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
        # Entrenar modelos de regresión
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
    ])
    
    # Pipeline de clasificación con clusters
    classification_with_clusters_pipeline = Pipeline([
        # Preparación de datos CON CLUSTERS
        node(
            func=prepare_classification_data_with_clusters,
            inputs=[
                "model_input_classification",
                "clustering_results", 
                "team_features_base"  # NUEVO INPUT
            ],
            outputs=[
                "X_train_clusters", 
                "X_test_clusters", 
                "y_train", 
                "y_test", 
                "feature_scaler_clusters",
                "feature_names_clusters"
            ],
            name="prepare_classification_data_with_clusters_node",
        ),
        # Entrenamiento de modelos con GridSearch
        node(
            func=train_classification_models,
            inputs=[
                "X_train_clusters", 
                "y_train", 
                "params:classification_models"
            ],
            outputs="trained_classification_models_with_clusters",
            name="train_classification_models_with_clusters_node",
        ),
        # Evaluación de modelos
        node(
            func=evaluate_classification_models,
            inputs=[
                "trained_classification_models_with_clusters", 
                "X_test_clusters", 
                "y_test", 
                "feature_scaler_clusters"
            ],
            outputs="classification_evaluation_with_clusters",
            name="evaluate_classification_models_with_clusters_node",
        ),
        # Análisis del mejor modelo
        node(
            func=analyze_best_model,
            inputs=[
                "classification_evaluation_with_clusters", 
                "X_test_clusters", 
                "y_test", 
                "feature_scaler_clusters"
            ],
            outputs="best_model_analysis_with_clusters",
            name="analyze_best_model_with_clusters_node",
        ),
        # Guardar resultados y modelos
        node(
            func=save_classification_results,
            inputs=[
                "trained_classification_models_with_clusters",
                "classification_evaluation_with_clusters", 
                "best_model_analysis_with_clusters",
                "feature_scaler_clusters",
                "params:classification_models"
            ],
            outputs=[
                "classification_report_with_clusters",
                "best_classification_model_with_clusters",
                "classification_scaler_with_clusters",
                "feature_importance_with_clusters"
            ],
            name="save_classification_results_with_clusters_node",
        ),
    ])
    
    # ========== PIPELINES COMBINADOS ==========
    
    # Pipeline completamente integrado CON CLUSTERING (RECOMENDADO)
    full_integrated_with_clusters_pipeline = (
        data_engineering_pipeline +
        clustering_pipeline +
        regression_with_clusters_pipeline +
        classification_with_clusters_pipeline
    )
    
    # Pipeline con solo clustering + clasificación
    clustering_classification_pipeline = (
        data_engineering_pipeline +
        clustering_pipeline +
        classification_with_clusters_pipeline
    )
    
    # Pipeline con solo clustering + regresión
    clustering_regression_pipeline = (
        data_engineering_pipeline +
        clustering_pipeline +
        regression_with_clusters_pipeline
    )
    
    # Pipeline solo con unsupervised learning
    unsupervised_pipeline = (
        data_engineering_pipeline +
        clustering_pipeline +
        dimensionality_pipeline +
        anomaly_pipeline
    )
    
    # Pipeline solo con supervised learning (original - sin clusters)
    supervised_pipeline = (
        data_engineering_pipeline +
        regression_pipeline +
        classification_pipeline
    )
    
    # Pipeline con integración básica (clustering + supervised original)
    integrated_basic_pipeline = (
        data_engineering_pipeline +
        clustering_pipeline +
        regression_pipeline +
        classification_pipeline
    )

    return {
        # ========== PIPELINE POR DEFECTO ==========
        "__default__": full_integrated_with_clusters_pipeline,
        
        # ========== PIPELINES INDIVIDUALES ==========
        "de": data_engineering_pipeline,
        "clf": classification_pipeline,
        "reg": regression_pipeline,
        "clustering": clustering_pipeline,
        "dimensionality": dimensionality_pipeline,
        "anomaly": anomaly_pipeline,
        
        # ========== PIPELINES CON INTEGRACIÓN DE CLUSTERING ==========
        "full_with_clusters": full_integrated_with_clusters_pipeline,
        "regression_with_clusters": regression_with_clusters_pipeline,
        "classification_with_clusters": classification_with_clusters_pipeline,
        "clustering_classification": clustering_classification_pipeline,
        "clustering_regression": clustering_regression_pipeline,
        
        # ========== PIPELINES ORIGINALES (SIN CLUSTERS) ==========
        "full_original": supervised_pipeline,
        "supervised_only": supervised_pipeline,
        
        # ========== PIPELINES ESPECIALIZADOS ==========
        "unsupervised_only": unsupervised_pipeline,
        "integrated_basic": integrated_basic_pipeline,
        
        # ========== PIPELINES PARA TESTING ==========
        "clustering_dimensionality": clustering_pipeline + dimensionality_pipeline,
        "anomaly_only": data_engineering_pipeline + anomaly_pipeline,
        "data_only": data_engineering_pipeline,
        "quick_test": data_engineering_pipeline + clustering_pipeline,
    }