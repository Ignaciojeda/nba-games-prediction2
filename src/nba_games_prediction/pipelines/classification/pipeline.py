"""
Pipeline para modelos de clasificación - Predicción de victorias NBA
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_classification_data,
    train_classification_models,
    evaluate_classification_models,
    analyze_best_model,
    save_classification_results
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # 1. Preparación de datos
            node(
                func=prepare_classification_data,
                inputs="model_input_classification",
                outputs=["X_train", "X_test", "y_train", "y_test", "feature_scaler"],
                name="prepare_classification_data_node",
                tags=["data_preparation", "classification"]
            ),
            
            # 2. Entrenamiento de modelos con GridSearch
            node(
                func=train_classification_models,
                inputs=["X_train", "y_train", "params:classification_models"],
                outputs="trained_classification_models",
                name="train_classification_models_node",
                tags=["model_training", "gridsearch", "classification"]
            ),
            
            # 3. Evaluación de modelos
            node(
                func=evaluate_classification_models,
                inputs=["trained_classification_models", "X_test", "y_test", "feature_scaler"],
                outputs="classification_evaluation_results",
                name="evaluate_classification_models_node",
                tags=["model_evaluation", "classification"]
            ),
            
            # 4. Análisis del mejor modelo
            node(
                func=analyze_best_model,
                inputs=["classification_evaluation_results", "X_test", "y_test", "feature_scaler"],
                outputs="best_model_analysis",
                name="analyze_best_model_node",
                tags=["model_analysis", "classification"]
            ),
            
            # 5. Guardar resultados y modelos
            node(
                func=save_classification_results,
                inputs=[
                    "trained_classification_models",
                    "classification_evaluation_results", 
                    "best_model_analysis",
                    "feature_scaler",
                    "params:classification_models"
                ],
                outputs=[
                    "classification_report",
                    "best_classification_model",
                    "classification_scaler",
                    "feature_importance"
                ],
                name="save_classification_results_node",
                tags=["model_persistence", "classification"]
            )
        ],
        tags="classification"
    )