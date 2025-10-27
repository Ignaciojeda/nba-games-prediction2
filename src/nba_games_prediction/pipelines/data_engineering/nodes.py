import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def clean_games_data(games: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza y transformación básica del dataset de partidos.
    """
    logger.info("Iniciando limpieza de datos de partidos")
    
    games_clean = games.copy()
    
    # Convertir fecha (ya se hace en load_args, pero por si acaso)
    games_clean['GAME_DATE_EST'] = pd.to_datetime(games_clean['GAME_DATE_EST'])
    
    # Extraer características temporales
    games_clean['YEAR'] = games_clean['GAME_DATE_EST'].dt.year
    games_clean['MONTH'] = games_clean['GAME_DATE_EST'].dt.month
    games_clean['DAY_OF_WEEK'] = games_clean['GAME_DATE_EST'].dt.dayofweek
    games_clean['SEASON'] = games_clean['SEASON'].astype(int)
    
    # Convertir variable objetivo a booleano
    games_clean['HOME_TEAM_WINS'] = games_clean['HOME_TEAM_WINS'].astype(bool)
    
    # Crear variables derivadas
    games_clean['POINT_DIFFERENTIAL'] = games_clean['PTS_home'] - games_clean['PTS_away']
    games_clean['TOTAL_POINTS'] = games_clean['PTS_home'] + games_clean['PTS_away']
    games_clean['MARGIN_OF_VICTORY'] = abs(games_clean['POINT_DIFFERENTIAL'])
    
    logger.info(f"Datos de partidos limpios: {games_clean.shape}")
    return games_clean

def clean_teams_data(teams: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza del dataset de equipos.
    """
    logger.info("Iniciando limpieza de datos de equipos")
    
    teams_clean = teams.copy()
    
    # Limpiar nombres de columnas (asegurar consistencia)
    teams_clean.columns = teams_clean.columns.str.upper()
    
    # Manejar valores missing
    teams_clean = teams_clean.fillna({
        'CITY': 'Unknown',
        'YEARFOUNDED': teams_clean['YEARFOUNDED'].median()
    })
    
    # Crear nombre completo del equipo
    teams_clean['FULL_NAME'] = teams_clean['CITY'] + ' ' + teams_clean['NICKNAME']
    
    logger.info(f"Datos de equipos limpios: {teams_clean.shape}")
    return teams_clean

def handle_missing_values(games_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Manejo de valores missing en el dataset de partidos.
    """
    logger.info("Manejando valores missing")
    
    games_no_missing = games_clean.copy()
    missing_cols = games_no_missing.columns[games_no_missing.isnull().any()].tolist()
    
    if missing_cols:
        logger.warning(f"Columnas con valores missing: {missing_cols}")
        
        # Estrategias diferentes por tipo de dato
        numeric_cols = games_no_missing.select_dtypes(include=[np.number]).columns
        categorical_cols = games_no_missing.select_dtypes(include=['object', 'category']).columns
        
        # Imputar numéricos con mediana
        for col in numeric_cols:
            if col in missing_cols:
                games_no_missing[col] = games_no_missing[col].fillna(games_no_missing[col].median())
        
        # Imputar categóricos con moda
        for col in categorical_cols:
            if col in missing_cols:
                games_no_missing[col] = games_no_missing[col].fillna(games_no_missing[col].mode()[0])
    
    logger.info("Valores missing manejados correctamente")
    return games_no_missing

def create_team_features_base(games_clean: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    Crear features base de rendimiento por equipo.
    """
    logger.info("Creando features base de rendimiento por equipo")
    
    # Rendimiento como local
    home_performance = games_clean.groupby(['HOME_TEAM_ID', 'SEASON']).agg({
        'PTS_home': ['mean', 'std', 'count'],
        'FG_PCT_home': 'mean',
        'FT_PCT_home': 'mean',
        'FG3_PCT_home': 'mean',
        'REB_home': 'mean',
        'AST_home': 'mean',
        'HOME_TEAM_WINS': 'mean',
        'POINT_DIFFERENTIAL': 'mean'
    }).reset_index()
    
    home_performance.columns = [
        'TEAM_ID', 'SEASON', 'HOME_PTS_MEAN', 'HOME_PTS_STD', 'HOME_GAMES_COUNT',
        'HOME_FG_PCT', 'HOME_FT_PCT', 'HOME_FG3_PCT', 'HOME_REB_MEAN', 'HOME_AST_MEAN',
        'HOME_WIN_PCT', 'HOME_PT_DIFF_MEAN'
    ]
    
    # Rendimiento como visitante
    away_performance = games_clean.groupby(['VISITOR_TEAM_ID', 'SEASON']).agg({
        'PTS_away': ['mean', 'std', 'count'],
        'FG_PCT_away': 'mean',
        'FT_PCT_away': 'mean',
        'FG3_PCT_away': 'mean',
        'REB_away': 'mean',
        'AST_away': 'mean',
        'HOME_TEAM_WINS': lambda x: (1 - x.mean()),  # Win % visitante
        'POINT_DIFFERENTIAL': lambda x: (-x.mean())  # Diferencia desde perspectiva visitante
    }).reset_index()
    
    away_performance.columns = [
        'TEAM_ID', 'SEASON', 'AWAY_PTS_MEAN', 'AWAY_PTS_STD', 'AWAY_GAMES_COUNT',
        'AWAY_FG_PCT', 'AWAY_FT_PCT', 'AWAY_FG3_PCT', 'AWAY_REB_MEAN', 'AWAY_AST_MEAN',
        'AWAY_WIN_PCT', 'AWAY_PT_DIFF_MEAN'
    ]
    
    # Combinar rendimiento local y visitante
    team_features = pd.merge(home_performance, away_performance, on=['TEAM_ID', 'SEASON'])
    
    # Calcular ventaja de local
    team_features['HOME_ADVANTAGE_PTS'] = team_features['HOME_PTS_MEAN'] - team_features['AWAY_PTS_MEAN']
    team_features['HOME_ADVANTAGE_WIN'] = team_features['HOME_WIN_PCT'] - team_features['AWAY_WIN_PCT']
    team_features['HOME_ADVANTAGE_REB'] = team_features['HOME_REB_MEAN'] - team_features['AWAY_REB_MEAN']
    
    # Unir con información de equipos
    team_features = pd.merge(team_features, teams[['TEAM_ID', 'NICKNAME', 'CITY']], 
                            on='TEAM_ID', how='left')
    
    logger.info(f"Features base de equipos creados: {team_features.shape}")
    return team_features

def create_game_level_features(games_clean: pd.DataFrame, team_features_base: pd.DataFrame) -> pd.DataFrame:
    """
    Crear features a nivel de partido para modelado.
    """
    logger.info("Creando features a nivel de partido")
    
    game_features = games_clean.copy()
    
    # Agregar features históricos del equipo local
    home_features = team_features_base.rename(columns={
        col: 'HOME_HIST_' + col for col in team_features_base.columns 
        if col not in ['TEAM_ID', 'SEASON', 'NICKNAME', 'CITY']
    })
    
    game_features = pd.merge(game_features, home_features, 
                           left_on=['HOME_TEAM_ID', 'SEASON'], 
                           right_on=['TEAM_ID', 'SEASON'], 
                           how='left')
    
    # Agregar features históricos del equipo visitante
    away_features = team_features_base.rename(columns={
        col: 'AWAY_HIST_' + col for col in team_features_base.columns 
        if col not in ['TEAM_ID', 'SEASON', 'NICKNAME', 'CITY']
    })
    
    game_features = pd.merge(game_features, away_features, 
                           left_on=['VISITOR_TEAM_ID', 'SEASON'], 
                           right_on=['TEAM_ID', 'SEASON'], 
                           how='left')
    
    # Crear features comparativas
    game_features['PTS_DIFF_HIST'] = (
        game_features['HOME_HIST_HOME_PTS_MEAN'] - game_features['AWAY_HIST_AWAY_PTS_MEAN']
    )
    game_features['WIN_PCT_DIFF'] = (
        game_features['HOME_HIST_HOME_WIN_PCT'] - game_features['AWAY_HIST_AWAY_WIN_PCT']
    )
    
    # Eliminar columnas duplicadas
    cols_to_drop = ['TEAM_ID_x', 'TEAM_ID_y', 'SEASON_x', 'SEASON_y', 
                   'NICKNAME_x', 'NICKNAME_y', 'CITY_x', 'CITY_y']
    game_features = game_features.drop(columns=[col for col in cols_to_drop if col in game_features.columns])
    
    logger.info(f"Features a nivel de partido creados: {game_features.shape}")
    return game_features

def prepare_model_inputs(game_level_features: pd.DataFrame, parameters: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preparar datasets para modelado de clasificación y regresión.
    
    Args:
        game_level_features: DataFrame con features a nivel de partido
        parameters: Parámetros de configuración
        
    Returns:
        Tuple con datasets para clasificación y regresión
    """
    logger.info("Preparando inputs para modelado")
    
    # Usar parámetros para seleccionar features
    feature_columns = parameters.get("features_clasificacion", [
        'PTS_home', 'PTS_away', 'FG_PCT_home', 'FG_PCT_away',
        'FT_PCT_home', 'FT_PCT_away', 'FG3_PCT_home', 'FG3_PCT_away',
        'REB_home', 'REB_away', 'AST_home', 'AST_away',
        'HOME_HIST_HOME_WIN_PCT', 'AWAY_HIST_AWAY_WIN_PCT', 'WIN_PCT_DIFF'
    ])
    
    classification_target = parameters.get("target_clasificacion", "HOME_TEAM_WINS")
    regression_target = "POINT_DIFFERENTIAL"
    
    # Filtrar columnas existentes
    available_features = [col for col in feature_columns if col in game_level_features.columns]
    
    # Dataset para clasificación
    classification_data = game_level_features[available_features + [classification_target]].copy()
    classification_data = classification_data.dropna()
    
    # Dataset para regresión
    regression_data = game_level_features[available_features + [regression_target]].copy()
    regression_data = regression_data.dropna()
    
    logger.info(f"Dataset clasificación: {classification_data.shape}")
    logger.info(f"Dataset regresión: {regression_data.shape}")
    
    return classification_data, regression_data

def validate_data_quality(games_clean: pd.DataFrame, 
                         team_features_base: pd.DataFrame,
                         game_level_features: pd.DataFrame) -> Dict[str, any]:
    """
    Validar la calidad de los datos procesados.
    """
    logger.info("Validando calidad de datos")
    
    validation_results = {}
    
    # Validar que no hay valores missing (convertir a int para JSON)
    validation_results['no_missing_games'] = int(games_clean.isnull().sum().sum() == 0)
    validation_results['no_missing_team_features'] = int(team_features_base.isnull().sum().sum() == 0)
    validation_results['no_missing_game_features'] = int(game_level_features.isnull().sum().sum() == 0)
    
    # Validar que los DataFrames no están vacíos (convertir a int)
    validation_results['games_not_empty'] = int(len(games_clean) > 0)
    validation_results['team_features_not_empty'] = int(len(team_features_base) > 0)
    validation_results['game_features_not_empty'] = int(len(game_level_features) > 0)
    
    # Validar tipos de datos (convertir a int)
    validation_results['correct_dtypes'] = int(all([
        games_clean['HOME_TEAM_WINS'].dtype == bool,
        pd.api.types.is_datetime64_any_dtype(games_clean['GAME_DATE_EST'])
    ]))
    
    # Agregar estadísticas adicionales para el reporte
    validation_results['summary'] = {
        'total_games': len(games_clean),
        'total_teams': len(team_features_base['TEAM_ID'].unique()),
        'total_features': game_level_features.shape[1],
        'validation_passed': all([
            validation_results['no_missing_games'],
            validation_results['no_missing_team_features'], 
            validation_results['no_missing_game_features'],
            validation_results['games_not_empty'],
            validation_results['team_features_not_empty'],
            validation_results['game_features_not_empty'],
            validation_results['correct_dtypes']
        ])
    }
    
    # Log validation results
    for test_name, result in validation_results.items():
        if test_name != 'summary':
            status = "PASS" if result else "FAIL"
            logger.info(f"Validación {test_name}: {status}")
    
    if validation_results['summary']['validation_passed']:
        logger.info("✅ Todas las validaciones pasaron correctamente")
    else:
        logger.warning("⚠️ Algunas validaciones fallaron")
    
    return validation_results

def create_eda_visualizations(games_clean: pd.DataFrame) -> Dict[str, plt.Figure]:
    """
    Crear visualizaciones para EDA.
    """
    logger.info("Creando visualizaciones EDA")
    
    figures = {}
    
    # 1. Distribución de puntos
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.histplot(games_clean["PTS_home"], kde=True, label="Local", color="blue", alpha=0.7, ax=ax1)
    sns.histplot(games_clean["PTS_away"], kde=True, label="Visitante", color="red", alpha=0.7, ax=ax1)
    ax1.set_title("Distribución de Puntos - Local vs Visitante")
    ax1.set_xlabel("Puntos")
    ax1.set_ylabel("Frecuencia")
    ax1.legend()
    figures["points_distribution"] = fig1
    
    # 2. Porcentaje de victorias por temporada
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    win_pct_by_season = games_clean.groupby('SEASON')['HOME_TEAM_WINS'].mean() * 100
    win_pct_by_season.plot(kind='bar', ax=ax2, color='green')
    ax2.set_title("Porcentaje de Victorias Locales por Temporada")
    ax2.set_xlabel("Temporada")
    ax2.set_ylabel("Porcentaje de Victorias (%)")
    ax2.tick_params(axis='x', rotation=45)
    figures["win_pct_by_season"] = fig2
    
    # 3. Correlación entre variables
    fig3, ax3 = plt.subplots(figsize=(14, 12))
    numeric_cols = games_clean.select_dtypes(include=[np.number]).columns
    correlation_matrix = games_clean[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax3)
    ax3.set_title("Matriz de Correlación - Variables Numéricas")
    figures["correlation_matrix"] = fig3
    
    logger.info(f"Visualizaciones EDA creadas: {len(figures)} figuras")
    return figures

def clean_games_details_data(games_details: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza del dataset de detalles de partidos.
    """
    logger.info("Iniciando limpieza de datos de detalles de partidos")
    
    games_details_clean = games_details.copy()
    
    # Manejar valores missing
    numeric_cols = games_details_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        games_details_clean[col] = games_details_clean[col].fillna(0)
    
    # Limpiar nombres de columnas
    games_details_clean.columns = games_details_clean.columns.str.upper()
    
    logger.info(f"Datos de detalles limpios: {games_details_clean.shape}")
    return games_details_clean

def create_games_teams_details(games_clean: pd.DataFrame, teams_clean: pd.DataFrame, games_details_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Crear dataset unificado de partidos, equipos y detalles.
    """
    logger.info("Creando dataset unificado games_teams_details")
    
    # Unir games con teams para información del equipo local
    games_with_home_teams = games_clean.merge(
        teams_clean[['TEAM_ID', 'NICKNAME', 'CITY']],
        left_on='HOME_TEAM_ID',
        right_on='TEAM_ID',
        suffixes=('', '_HOME')
    )
    
    # Unir con información del equipo visitante
    games_with_both_teams = games_with_home_teams.merge(
        teams_clean[['TEAM_ID', 'NICKNAME', 'CITY']],
        left_on='VISITOR_TEAM_ID', 
        right_on='TEAM_ID',
        suffixes=('_HOME', '_AWAY')
    )
    
    # Renombrar columnas para claridad
    games_with_both_teams = games_with_both_teams.rename(columns={
        'NICKNAME_HOME': 'HOME_TEAM_NAME',
        'CITY_HOME': 'HOME_TEAM_CITY',
        'NICKNAME_AWAY': 'AWAY_TEAM_NAME', 
        'CITY_AWAY': 'AWAY_TEAM_CITY'
    })
    
    # Unir con games_details (si está disponible y no está vacío)
    if not games_details_clean.empty and len(games_details_clean) > 0:
        try:
            # Tomar solo algunas columnas relevantes de games_details
            details_subset = games_details_clean[['GAME_ID', 'TEAM_ID', 'PLAYER_NAME', 'PTS', 'REB', 'AST']].drop_duplicates()
            
            # Agregar estadísticas por equipo del game_details
            team_stats_from_details = details_subset.groupby(['GAME_ID', 'TEAM_ID']).agg({
                'PTS': 'sum',
                'REB': 'sum', 
                'AST': 'sum',
                'PLAYER_NAME': 'count'  # Número de jugadores
            }).reset_index()
            
            team_stats_from_details.columns = ['GAME_ID', 'TEAM_ID', 'DETAIL_PTS', 'DETAIL_REB', 'DETAIL_AST', 'PLAYER_COUNT']
            
            # Unir estadísticas del equipo local
            final_dataset = games_with_both_teams.merge(
                team_stats_from_details,
                left_on=['GAME_ID', 'HOME_TEAM_ID'],
                right_on=['GAME_ID', 'TEAM_ID'],
                how='left',
                suffixes=('', '_HOME_DETAILS')
            )
            
            # Unir estadísticas del equipo visitante  
            final_dataset = final_dataset.merge(
                team_stats_from_details,
                left_on=['GAME_ID', 'VISITOR_TEAM_ID'],
                right_on=['GAME_ID', 'TEAM_ID'],
                how='left',
                suffixes=('_HOME', '_AWAY')
            )
            
        except Exception as e:
            logger.warning(f"No se pudieron unir game details: {e}")
            final_dataset = games_with_both_teams
    else:
        logger.info("games_details_clean está vacío, continuando sin detalles de jugadores")
        final_dataset = games_with_both_teams
    
    # Limpiar columnas duplicadas
    cols_to_drop = ['TEAM_ID_HOME', 'TEAM_ID_AWAY', 'TEAM_ID', 'TEAM_ID_HOME', 'TEAM_ID_AWAY']
    final_dataset = final_dataset.drop(columns=[col for col in cols_to_drop if col in final_dataset.columns])
    
    logger.info(f"Dataset games_teams_details creado: {final_dataset.shape}")
    return final_dataset