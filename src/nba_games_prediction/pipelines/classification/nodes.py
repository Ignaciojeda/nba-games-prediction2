"""
Nodes para pipeline de clasificación
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import logging
import json
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def prepare_classification_data(classification_data: pd.DataFrame) -> Tuple:
    """
    Preparar datos para clasificación
    """
    logger.info("Preparando datos para clasificación")
    
    # Separar features y target
    X = classification_data.drop('HOME_TEAM_WINS', axis=1)
    y = classification_data['HOME_TEAM_WINS']
    
    logger.info(f"Features: {X.shape[1]}, Target: {y.shape[0]}")
    logger.info(f"Distribución del target: {y.value_counts().to_dict()}")
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("✅ Datos preparados para clasificación")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def _instantiate_model(model_config: Dict) -> Any:
    """Instanciar modelo desde configuración"""
    # Importar la clase del modelo
    if model_config['class'] == 'sklearn.linear_model.LogisticRegression':
        model_class = LogisticRegression
    elif model_config['class'] == 'sklearn.tree.DecisionTreeClassifier':
        model_class = DecisionTreeClassifier
    elif model_config['class'] == 'sklearn.ensemble.RandomForestClassifier':
        model_class = RandomForestClassifier
    elif model_config['class'] == 'sklearn.svm.SVC':
        model_class = SVC
    elif model_config['class'] == 'sklearn.naive_bayes.GaussianNB':
        model_class = GaussianNB
    else:
        raise ValueError(f"Clase de modelo no reconocida: {model_config['class']}")
    
    # Obtener parámetros
    model_kwargs = model_config.get('kwargs', {})
    return model_class(**model_kwargs)

def train_classification_models(X_train: np.ndarray, y_train: pd.Series, 
                              models_config: Dict) -> Dict[str, Any]:
    """
    Entrenar modelos de clasificación con GridSearchCV
    """
    logger.info("Iniciando entrenamiento de modelos de clasificación")
    
    trained_models = {}
    
    for model_name, config in models_config.items():
        logger.info(f"Entrenando {model_name}...")
        
        # Instanciar modelo
        model = _instantiate_model(config['model'])
        
        if config.get('params', {}):  # Si tiene parámetros para GridSearch
            grid_search = GridSearchCV(
                model, 
                config['params'], 
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
        else:  # Sin GridSearch (Naive Bayes)
            model.fit(X_train, y_train)
            best_model = model
            best_params = "No GridSearch"
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        
        trained_models[model_name] = {
            'model': best_model,
            'best_params': best_params,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        logger.info(f"✅ {model_name} - CV Accuracy: {cv_scores.mean():.4f}")
    
    return trained_models

def evaluate_classification_models(trained_models: Dict, X_test: np.ndarray, 
                                 y_test: pd.Series, scaler: StandardScaler) -> Dict:
    """
    Evaluar modelos en conjunto de test
    """
    logger.info("Evaluando modelos en conjunto de test")
    
    evaluation_results = {}
    
    for model_name, model_info in trained_models.items():
        model = model_info['model']
        
        # Predecir
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Reporte de clasificación
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        evaluation_results[model_name] = {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
        }
        
        # CORRECCIÓN: Manejar el caso cuando auc_roc es None
        if auc_roc is not None:
            logger.info(f"📊 {model_name} - Test Accuracy: {accuracy:.4f}, AUC-ROC: {auc_roc:.4f}")
        else:
            logger.info(f"📊 {model_name} - Test Accuracy: {accuracy:.4f}, AUC-ROC: N/A")
    
    return evaluation_results

def analyze_best_model(evaluation_results: Dict, X_test: np.ndarray, 
                      y_test: pd.Series, scaler: StandardScaler) -> Dict:
    """
    Analizar el mejor modelo y generar insights
    """
    logger.info("Analizando mejor modelo")
    
    # Encontrar mejor modelo por accuracy
    best_model_name = max(evaluation_results.keys(), 
                         key=lambda x: evaluation_results[x]['accuracy'])
    
    best_model_metrics = evaluation_results[best_model_name]
    
    analysis = {
        'best_model_name': best_model_name,
        'best_accuracy': best_model_metrics['accuracy'],
        'best_auc_roc': best_model_metrics['auc_roc'],
        'baseline_accuracy': y_test.mean(),  # Accuracy del modelo base
        'improvement_absolute': best_model_metrics['accuracy'] - y_test.mean(),
        'improvement_relative': (best_model_metrics['accuracy'] - y_test.mean()) / y_test.mean() * 100
    }
    
    logger.info(f"🏆 Mejor modelo: {best_model_name}")
    logger.info(f"📈 Accuracy: {best_model_metrics['accuracy']:.4f}")
    logger.info(f"📊 Mejora vs baseline: {analysis['improvement_relative']:.1f}%")
    
    return analysis

def save_classification_results(trained_models: Dict, evaluation_results: Dict,
                              best_model_analysis: Dict, scaler: StandardScaler,
                              models_config: Dict) -> Tuple:
    """
    Guardar resultados, modelos y generar reporte
    """
    logger.info("Guardando resultados de clasificación")
    
    best_model_name = best_model_analysis['best_model_name']
    best_model = trained_models[best_model_name]['model']
    
    # Feature importance si está disponible (SVM no lo tiene)
    feature_importance = {}
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = {
            'feature_names': list(range(len(best_model.feature_importances_))),
            'importance_scores': best_model.feature_importances_.tolist(),
            'has_importance': True
        }
        logger.info(f"🔍 Feature importance disponible para {best_model_name}")
    else:
        feature_importance = {
            'has_importance': False,
            'message': f"Modelo {best_model_name} no tiene feature_importances_"
        }
        logger.info(f"⚠️ Feature importance no disponible para {best_model_name}")
    
    # Crear reporte consolidado
    classification_report = {
        'models_trained': list(trained_models.keys()),
        'evaluation_results': evaluation_results,
        'best_model_analysis': best_model_analysis,
        'training_config': models_config,
        'summary': {
            'best_model': best_model_name,
            'best_accuracy': best_model_analysis['best_accuracy'],
            'baseline_accuracy': best_model_analysis['baseline_accuracy'],
            'improvement': best_model_analysis['improvement_relative'],
            'best_auc_roc': best_model_analysis['best_auc_roc']
        }
    }
    
    logger.info("✅ Resultados de clasificación guardados")
    
    return classification_report, best_model, scaler, feature_importance