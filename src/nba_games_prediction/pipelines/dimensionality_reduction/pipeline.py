from kedro.pipeline import Pipeline, node
from .nodes import (
    perform_pca_analysis,
    perform_tsne_analysis,
    perform_umap_analysis,
    compare_dimensionality_methods,
    integrate_dimensionality_features
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # 1. Análisis PCA completo
            node(
                func=perform_pca_analysis,
                inputs=["clustering_data", "params:dimensionality_reduction"],
                outputs="pca_results",
                name="pca_analysis_node"
            ),
            
            # 2. Análisis t-SNE
            node(
                func=perform_tsne_analysis,
                inputs=["clustering_data", "params:dimensionality_reduction"],
                outputs="tsne_results", 
                name="tsne_analysis_node"
            ),
            
            # 3. Análisis UMAP
            node(
                func=perform_umap_analysis,
                inputs=["clustering_data", "params:dimensionality_reduction"],
                outputs="umap_results",
                name="umap_analysis_node"
            ),
            
            # 4. Comparación de métodos
            node(
                func=compare_dimensionality_methods,
                inputs=["pca_results", "tsne_results", "umap_results"],
                outputs="dimensionality_comparison",
                name="dimensionality_comparison_node"
            ),
            
            # 5. Integración con features existentes
            node(
                func=integrate_dimensionality_features,
                inputs=["pca_results", "team_features_base", "params:dimensionality_reduction"],
                outputs="enhanced_features_with_pca",
                name="integrate_dimensionality_features_node"
            ),
        ]
    )