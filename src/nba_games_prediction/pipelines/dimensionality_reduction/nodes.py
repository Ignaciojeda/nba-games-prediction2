import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import logging

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def perform_pca_analysis(clustering_data: pd.DataFrame, params: Dict) -> Dict:
    """Análisis PCA completo con varianza explicada, loadings y biplot."""
    logger.info("Realizando análisis PCA completo")
    
    n_components = params.get("pca_n_components", 10)
    random_state = params.get("random_state", 42)
    
    # Estandarizar datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(clustering_data)
    
    # Aplicar PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    principal_components = pca.fit_transform(data_scaled)
    
    # Calcular varianza explicada
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Obtener loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Crear DataFrame de componentes principales
    pc_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
    principal_df = pd.DataFrame(principal_components, columns=pc_columns, index=clustering_data.index)
    
    # Identificar features más importantes por componente
    feature_importance = {}
    for i, pc in enumerate(pc_columns):
        # Tomar las 5 features con mayor peso absoluto
        feature_weights = pd.Series(loadings[:, i], index=clustering_data.columns)
        top_features = feature_weights.abs().nlargest(5)
        feature_importance[pc] = top_features.to_dict()
    
    pca_results = {
        'pca_model': pca,
        'principal_components': principal_df,
        'explained_variance': explained_variance.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'loadings': loadings.tolist(),
        'feature_importance': feature_importance,
        'n_components': n_components,
        'total_variance_explained': cumulative_variance[-1],
        'scaler': scaler
    }
    
    logger.info(f"PCA completado: {n_components} componentes explican {cumulative_variance[-1]:.1%} de varianza")
    
    return pca_results

def perform_tsne_analysis(clustering_data: pd.DataFrame, params: Dict) -> Dict:
    """Análisis t-SNE para visualización de alta dimensión."""
    logger.info("Realizando análisis t-SNE")
    
    n_components = params.get("tsne_n_components", 2)
    perplexity = params.get("tsne_perplexity", 30)
    random_state = params.get("random_state", 42)
    
    # Estandarizar datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(clustering_data)
    
    # Aplicar t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000
    )
    
    tsne_components = tsne.fit_transform(data_scaled)
    
    # Crear DataFrame
    tsne_columns = [f'TSNE_{i+1}' for i in range(n_components)]
    tsne_df = pd.DataFrame(tsne_components, columns=tsne_columns, index=clustering_data.index)
    
    tsne_results = {
        'tsne_model': tsne,
        'tsne_components': tsne_df,
        'perplexity': perplexity,
        'n_components': n_components,
        'scaler': scaler
    }
    
    logger.info("Análisis t-SNE completado")
    
    return tsne_results

def perform_umap_analysis(clustering_data: pd.DataFrame, params: Dict) -> Dict:
    """Análisis UMAP como alternativa moderna a t-SNE."""
    logger.info("Realizando análisis UMAP")
    
    n_components = params.get("umap_n_components", 2)
    n_neighbors = params.get("umap_n_neighbors", 15)
    min_dist = params.get("umap_min_dist", 0.1)
    random_state = params.get("random_state", 42)
    
    # Estandarizar datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(clustering_data)
    
    # Aplicar UMAP
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    
    umap_components = umap_model.fit_transform(data_scaled)
    
    # Crear DataFrame
    umap_columns = [f'UMAP_{i+1}' for i in range(n_components)]
    umap_df = pd.DataFrame(umap_components, columns=umap_columns, index=clustering_data.index)
    
    umap_results = {
        'umap_model': umap_model,
        'umap_components': umap_df,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'n_components': n_components,
        'scaler': scaler
    }
    
    logger.info("Análisis UMAP completado")
    
    return umap_results

def compare_dimensionality_methods(pca_results: Dict, tsne_results: Dict, umap_results: Dict) -> Dict:
    """Comparar los diferentes métodos de reducción dimensional."""
    logger.info("Comparando métodos de reducción dimensional")
    
    comparison = {
        'pca': {
            'n_components': pca_results['n_components'],
            'variance_explained': pca_results['total_variance_explained'],
            'type': 'Linear',
            'preserves_global_structure': True,
            'interpretability': 'High'
        },
        'tsne': {
            'n_components': tsne_results['n_components'],
            'perplexity': tsne_results['perplexity'],
            'type': 'Non-linear',
            'preserves_local_structure': True,
            'interpretability': 'Medium'
        },
        'umap': {
            'n_components': umap_results['n_components'],
            'n_neighbors': umap_results['n_neighbors'],
            'type': 'Non-linear',
            'preserves_both_structure': True,
            'interpretability': 'Medium'
        }
    }
    
    # Recomendación basada en el uso
    recommendations = [
        "PCA: Mejor para análisis exploratorio y feature engineering",
        "t-SNE: Mejor para visualización y clustering",
        "UMAP: Balance entre preservación de estructura y velocidad"
    ]
    
    comparison['recommendations'] = recommendations
    
    logger.info("Comparación de métodos completada")
    
    return comparison

def integrate_dimensionality_features(pca_results: Dict, team_features_base: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Integrar componentes PCA como nuevas features para modelos supervisados."""
    logger.info("Integrando componentes PCA como features")
    
    # Obtener componentes principales
    principal_components = pca_results['principal_components']
    
    # Unir con features originales
    enhanced_features = team_features_base.copy()
    
    # Agregar componentes principales
    for col in principal_components.columns:
        enhanced_features[col] = principal_components[col]
    
    # Agregar metadata de PCA
    enhanced_features['PCA_VARIANCE_EXPLAINED'] = pca_results['total_variance_explained']
    
    logger.info(f"Features mejoradas con PCA: {enhanced_features.shape}")
    
    return enhanced_features