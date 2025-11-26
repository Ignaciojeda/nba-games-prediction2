"""
Pipeline para clasificación con integración de clusters - Predicción de victorias NBA
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_classification_data_with_clusters,
    train_classification_models,
    evaluate_classification_models,
    analyze_best_model,
    save_classification_results
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # 1. Preparación de datos CON CLUSTERS
            node(
                func=prepare_classification_data_with_clusters,
                inputs=[
                    "model_input_classification",
                    "clustering_results", 
                    "team_features_base"
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
                tags=["data_preparation", "classification", "clustering_integration"]
            ),
            
            # 2. Entrenamiento de modelos con GridSearch
            node(
                func=train_classification_models,
                inputs=[
                    "X_train_clusters", 
                    "y_train", 
                    "params:classification_models"
                ],
                outputs="trained_classification_models_with_clusters",
                name="train_classification_models_with_clusters_node",
                tags=["model_training", "gridsearch", "classification", "clustering"]
            ),
            
            # 3. Evaluación de modelos
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
                tags=["model_evaluation", "classification", "clustering"]
            ),
            
            # 4. Análisis del mejor modelo
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
                tags=["model_analysis", "classification", "clustering"]
            ),
            
            # 5. Guardar resultados y modelos
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
                tags=["model_persistence", "classification", "clustering"]
            )
        ],
        tags="classification_with_clusters"
    )