"""
Pipeline para detección de anomalías - NBA Prediction
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    detect_anomalies_isolation_forest,
    detect_anomalies_lof,
    detect_anomalies_one_class_svm,
    analyze_anomalies,
    integrate_anomaly_scores
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # 1. Isolation Forest
            node(
                func=detect_anomalies_isolation_forest,
                inputs=["clustering_data", "params:anomaly_detection"],
                outputs="isolation_forest_anomalies",
                name="isolation_forest_anomalies_node"
            ),
            
            # 2. Local Outlier Factor (LOF)
            node(
                func=detect_anomalies_lof,
                inputs=["clustering_data", "params:anomaly_detection"],
                outputs="lof_anomalies",
                name="lof_anomalies_node"
            ),
            
            # 3. One-Class SVM
            node(
                func=detect_anomalies_one_class_svm,
                inputs=["clustering_data", "params:anomaly_detection"],
                outputs="one_class_svm_anomalies",
                name="one_class_svm_anomalies_node"
            ),
            
            # 4. Análisis comparativo de anomalías
            node(
                func=analyze_anomalies,
                inputs=[
                    "isolation_forest_anomalies", 
                    "lof_anomalies", 
                    "one_class_svm_anomalies", 
                    "team_features_base"
                ],
                outputs="anomaly_analysis_report",
                name="analyze_anomalies_node"
            ),
            
            # 5. Integrar scores de anomalías como features
            node(
                func=integrate_anomaly_scores,
                inputs=["isolation_forest_anomalies", "team_features_base"],
                outputs="features_with_anomaly_scores",
                name="integrate_anomaly_scores_node"
            ),
        ],
        tags="anomaly_detection"
    )