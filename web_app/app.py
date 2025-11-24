import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="NBA Games Prediction",
    page_icon="🏀",
    layout="wide"
)

# Título principal
st.title("🏀 NBA Games Prediction Dashboard")
st.markdown("---")

# Configurar rutas
BASE_PATH = Path("..")
DATA_PATH = BASE_PATH / "data"
PRIMARY_PATH = DATA_PATH / "03_primary"
MODELS_PATH = DATA_PATH / "06_models"
RAW_PATH = DATA_PATH / "01_raw"

# Configuración para Plotly
plotly_config = {
    'displayModeBar': True,
    'displaylogo': False
}

# Función segura para cargar pickle
def safe_load_pickle(file_path):
    """Cargar archivo pickle con manejo de errores"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

# Función para cargar nombres de equipos
@st.cache_data
def load_team_names():
    """Cargar mapeo de ID a nombres de equipos"""
    try:
        # Intentar cargar desde raw_teams
        teams_path = RAW_PATH / "raw_teams.csv"
        if teams_path.exists():
            teams_df = pd.read_csv(teams_path)
            if 'TEAM_ID' in teams_df.columns and 'NICKNAME' in teams_df.columns:
                return dict(zip(teams_df['TEAM_ID'], teams_df['NICKNAME']))
            elif 'TEAM_ID' in teams_df.columns and 'ABBREVIATION' in teams_df.columns:
                return dict(zip(teams_df['TEAM_ID'], teams_df['ABBREVIATION']))
        
        # Si no funciona, crear un mapeo básico
        basic_mapping = {
            1610612737: 'Hawks', 1610612738: 'Celtics', 1610612739: 'Cavaliers',
            1610612740: 'Pelicans', 1610612741: 'Bulls', 1610612742: 'Mavericks',
            1610612743: 'Nuggets', 1610612744: 'Warriors', 1610612745: 'Rockets',
            1610612746: 'Clippers', 1610612747: 'Lakers', 1610612748: 'Heat',
            1610612749: 'Bucks', 1610612750: 'Timberwolves', 1610612751: 'Nets',
            1610612752: 'Knicks', 1610612753: 'Magic', 1610612754: 'Pacers',
            1610612755: '76ers', 1610612756: 'Suns', 1610612757: 'Blazers',
            1610612758: 'Kings', 1610612759: 'Spurs', 1610612760: 'Thunder',
            1610612761: 'Raptors', 1610612762: 'Jazz', 1610612763: 'Grizzlies',
            1610612764: 'Wizards', 1610612765: 'Pistons', 1610612766: 'Hornets'
        }
        return basic_mapping
    except Exception as e:
        st.sidebar.error(f"Error cargando nombres de equipos: {e}")
        return {}

# Función para cargar datos
@st.cache_data
def load_data():
    """Cargar todos los datos preprocesados"""
    data_dict = {}
    
    st.sidebar.info("📁 Cargando datos del pipeline...")
    
    try:
        # Cargar nombres de equipos
        team_names = load_team_names()
        data_dict['team_names'] = team_names
        if team_names:
            st.sidebar.success("✅ Nombres de equipos")
        
        # Cargar datos de equipos
        team_features_path = PRIMARY_PATH / "team_features_base.parquet"
        if team_features_path.exists():
            team_features = pd.read_parquet(team_features_path)
            # Agregar nombres de equipos si están disponibles
            if team_names:
                team_features['TEAM_NAME'] = team_features['TEAM_ID'].map(team_names)
            data_dict['team_features'] = team_features
            st.sidebar.success("✅ Datos de equipos")
        
        # Cargar datos de partidos
        game_features_path = PRIMARY_PATH / "game_level_features.parquet"
        if game_features_path.exists():
            game_features = pd.read_parquet(game_features_path)
            # Agregar nombres de equipos si están disponibles
            if team_names:
                # Buscar columnas que puedan contener TEAM_ID
                for col in game_features.columns:
                    if 'TEAM' in col and 'ID' in col:
                        game_features[f'{col}_NAME'] = game_features[col].map(team_names)
            data_dict['game_features'] = game_features
            st.sidebar.success("✅ Datos de partidos")
        
        # Cargar datos de clustering CSV
        clustering_data_path = MODELS_PATH / "clustering_data.csv"
        if clustering_data_path.exists():
            clustering_data = pd.read_csv(clustering_data_path)
            # Agregar nombres de equipos
            if team_names and 'TEAM_ID' in clustering_data.columns:
                clustering_data['TEAM_NAME'] = clustering_data['TEAM_ID'].map(team_names)
            data_dict['clustering_data'] = clustering_data
            st.sidebar.success("✅ Datos de clustering")
        
        # Cargar modelos de regresión
        regression_models = {}
        model_files = {
            'away_weakness': 'away_weakness_model.pkl',
            'local_strength': 'local_strength_model.pkl', 
            'point_differential': 'point_differential_model.pkl'
        }
        
        for model_name, file_name in model_files.items():
            model_path = MODELS_PATH / file_name
            if model_path.exists():
                model = safe_load_pickle(model_path)
                if model is not None:
                    regression_models[model_name] = model
                    st.sidebar.success(f"✅ Modelo {model_name}")
        
        data_dict['regression_models'] = regression_models
        
        return data_dict
        
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return data_dict

def show_prediction_forms(data_dict):
    """Formularios para predicciones"""
    st.header("🎯 Sistema de Predicción NBA")
    
    if 'team_features' not in data_dict:
        st.warning("No se encontraron datos de equipos para realizar predicciones.")
        return
    
    team_features = data_dict['team_features']
    regression_models = data_dict.get('regression_models', {})
    team_names = data_dict.get('team_names', {})
    
    # Pestañas para diferentes tipos de análisis
    tab1, tab2, tab3 = st.tabs([
        "📊 Equipos Peores como Visitantes", 
        "🎯 Predicción Ganador Local", 
        "📈 Predicción Métricas de Rendimiento"
    ])
    
    with tab1:
        show_away_weakness_analysis(team_features, team_names)
    
    with tab2:
        show_winner_prediction(team_features, team_names)
    
    with tab3:
        show_performance_prediction(team_features, regression_models, team_names)

def show_away_weakness_analysis(team_features, team_names):
    """Análisis de equipos peores como visitantes"""
    st.subheader("📌 Análisis: Equipos Peores como Visitantes")
    
    # Métricas para identificar equipos débiles como visitantes
    st.markdown("""
    **Criterios para identificar equipos débiles como visitantes:**
    - Bajo porcentaje de victorias fuera de casa
    - Diferencia negativa en puntos como visitante
    - Bajo rendimiento ofensivo fuera de casa
    """)
    
    # Calcular métricas de debilidad como visitante
    if 'AWAY_WIN_PCT' in team_features.columns and 'AWAY_PTS_MEAN' in team_features.columns:
        # Agrupar por equipo y calcular promedios
        team_away_stats = team_features.groupby('TEAM_ID').agg({
            'AWAY_WIN_PCT': 'mean',
            'AWAY_PTS_MEAN': 'mean',
            'HOME_WIN_PCT': 'mean',
            'HOME_PTS_MEAN': 'mean'
        }).round(3)
        
        # Agregar nombres de equipos
        if team_names:
            team_away_stats['TEAM_NAME'] = team_away_stats.index.map(team_names)
        
        # Calcular diferencia entre rendimiento en casa vs fuera
        team_away_stats['WIN_PCT_DIFF'] = team_away_stats['HOME_WIN_PCT'] - team_away_stats['AWAY_WIN_PCT']
        team_away_stats['PTS_DIFF'] = team_away_stats['HOME_PTS_MEAN'] - team_away_stats['AWAY_PTS_MEAN']
        
        # Ordenar por debilidad como visitante
        weakest_away = team_away_stats.nsmallest(10, 'AWAY_WIN_PCT')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏀 Top 10 Equipos Más Débiles como Visitantes")
            display_cols = ['AWAY_WIN_PCT', 'AWAY_PTS_MEAN', 'WIN_PCT_DIFF']
            if 'TEAM_NAME' in weakest_away.columns:
                display_df = weakest_away[['TEAM_NAME'] + display_cols]
                display_df.columns = ['Equipo', 'Win% Visitante', 'Puntos Promedio Visitante', 'Diferencia Win%']
            else:
                display_df = weakest_away[display_cols]
                display_df.columns = ['Win% Visitante', 'Puntos Promedio Visitante', 'Diferencia Win%']
            
            st.dataframe(display_df, width='stretch')
        
        with col2:
            # Gráfico de comparación
            fig = go.Figure()
            x_labels = weakest_away['TEAM_NAME'] if 'TEAM_NAME' in weakest_away.columns else weakest_away.index
            
            fig.add_trace(go.Bar(name='Win% Casa', 
                               x=x_labels, 
                               y=weakest_away['HOME_WIN_PCT'],
                               marker_color='lightblue'))
            fig.add_trace(go.Bar(name='Win% Visitante', 
                               x=x_labels, 
                               y=weakest_away['AWAY_WIN_PCT'],
                               marker_color='lightcoral'))
            fig.update_layout(title='Comparación: Rendimiento en Casa vs Fuera (Equipos Más Débiles)',
                            barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
        
        # Análisis detallado por equipo
        st.subheader("🔍 Análisis Detallado por Equipo")
        if 'TEAM_NAME' in weakest_away.columns:
            team_options = list(zip(weakest_away['TEAM_NAME'], weakest_away.index))
            selected_team_name, selected_team_id = st.selectbox(
                "Selecciona un equipo para análisis detallado:", 
                team_options,
                format_func=lambda x: x[0]
            )
        else:
            selected_team_id = st.selectbox("Selecciona un equipo para análisis detallado:", 
                                           weakest_away.index)
            selected_team_name = selected_team_id
        
        if selected_team_id:
            team_data = weakest_away.loc[selected_team_id]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Win% Visitante", f"{team_data['AWAY_WIN_PCT']:.3f}")
            with col2:
                st.metric("Win% Casa", f"{team_data['HOME_WIN_PCT']:.3f}")
            with col3:
                st.metric("Diferencia", f"{team_data['WIN_PCT_DIFF']:.3f}")
            with col4:
                st.metric("Puntos Visitante", f"{team_data['AWAY_PTS_MEAN']:.1f}")

def show_winner_prediction(team_features, team_names):
    """Predicción de ganador local"""
    st.subheader("🎯 Predicción: ¿Ganará el Equipo Local?")
    
    st.markdown("""
    **Simulador de Partido:**
    Selecciona dos equipos para analizar las probabilidades de victoria del equipo local
    basado en su rendimiento histórico.
    """)
    
    # Obtener lista de equipos con nombres
    if 'TEAM_NAME' in team_features.columns:
        team_options = list(zip(team_features['TEAM_NAME'].unique(), 
                              team_features['TEAM_ID'].unique()))
    else:
        team_options = [(str(team_id), team_id) for team_id in team_features['TEAM_ID'].unique()]
    
    # Selectores de equipos
    col1, col2 = st.columns(2)
    
    with col1:
        home_team_name, home_team_id = st.selectbox(
            "🏠 Equipo Local:", 
            team_options,
            format_func=lambda x: x[0],
            key="home_team"
        )
    
    with col2:
        away_team_name, away_team_id = st.selectbox(
            "✈️ Equipo Visitante:", 
            team_options,
            format_func=lambda x: x[0],
            key="away_team"
        )
    
    if home_team_id and away_team_id and home_team_id != away_team_id:
        # Obtener datos de los equipos
        home_data = team_features[team_features['TEAM_ID'] == home_team_id].iloc[0]
        away_data = team_features[team_features['TEAM_ID'] == away_team_id].iloc[0]
        
        # Calcular probabilidades simples basadas en rendimiento histórico
        if all(col in home_data for col in ['HOME_WIN_PCT', 'AWAY_WIN_PCT']):
            home_strength = home_data['HOME_WIN_PCT']  # Fuerza local del equipo local
            away_weakness = 1 - away_data['AWAY_WIN_PCT']  # Debilidad visitante del equipo away
            
            # Probabilidad simple (podría mejorarse con un modelo más complejo)
            home_win_probability = (home_strength + away_weakness) / 2
            
            # Mostrar resultados
            st.subheader("📊 Probabilidades de Victoria")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Probabilidad Victoria Local", f"{home_win_probability:.1%}")
            
            with col2:
                st.metric("Probabilidad Victoria Visitante", f"{(1 - home_win_probability):.1%}")
            
            with col3:
                if home_win_probability > 0.6:
                    st.metric("Recomendación", "✅ Fuerte favorito local")
                elif home_win_probability > 0.5:
                    st.metric("Recomendación", "⚖️ Ligero favorito local")
                else:
                    st.metric("Recomendación", "⚠️ Posible sorpresa visitante")
            
            # Comparativa de métricas
            st.subheader("📈 Comparativa de Equipos")
            
            comparison_data = pd.DataFrame({
                'Métrica': ['Win% en Casa', 'Win% como Visitante', 'Puntos en Casa', 'Puntos como Visitante'],
                home_team_name: [
                    home_data.get('HOME_WIN_PCT', 0),
                    home_data.get('AWAY_WIN_PCT', 0),
                    home_data.get('HOME_PTS_MEAN', 0),
                    home_data.get('AWAY_PTS_MEAN', 0)
                ],
                away_team_name: [
                    away_data.get('HOME_WIN_PCT', 0),
                    away_data.get('AWAY_WIN_PCT', 0),
                    away_data.get('HOME_PTS_MEAN', 0),
                    away_data.get('AWAY_PTS_MEAN', 0)
                ]
            })
            
            fig = px.bar(comparison_data, x='Métrica', y=[home_team_name, away_team_name],
                        title=f'Comparativa: {home_team_name} vs {away_team_name}', barmode='group')
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)

def show_performance_prediction(team_features, regression_models, team_names):
    """Predicción de métricas de rendimiento"""
    st.subheader("📈 Predicción de Métricas de Rendimiento")
    
    st.markdown("""
    **Simulador de Métricas:**
    Predice métricas específicas de rendimiento para un partido basado en modelos de regresión.
    """)
    
    # Obtener lista de equipos con nombres
    if 'TEAM_NAME' in team_features.columns:
        team_options = list(zip(team_features['TEAM_NAME'].unique(), 
                              team_features['TEAM_ID'].unique()))
    else:
        team_options = [(str(team_id), team_id) for team_id in team_features['TEAM_ID'].unique()]
    
    # Selector de equipos
    col1, col2 = st.columns(2)
    
    with col1:
        pred_home_team_name, pred_home_team_id = st.selectbox(
            "🏠 Equipo Local:", 
            team_options,
            format_func=lambda x: x[0],
            key="pred_home"
        )
    
    with col2:
        pred_away_team_name, pred_away_team_id = st.selectbox(
            "✈️ Equipo Visitante:", 
            team_options,
            format_func=lambda x: x[0],
            key="pred_away"
        )
    
    # Selector de métrica a predecir
    metric_options = {
        'away_weakness': 'Debilidad como Visitante',
        'local_strength': 'Fuerza Local', 
        'point_differential': 'Diferencia de Puntos'
    }
    
    selected_metric = st.selectbox("📊 Métrica a Predecir:", 
                                  list(metric_options.values()))
    
    if pred_home_team_id and pred_away_team_id and pred_home_team_id != pred_away_team_id:
        # Obtener datos de los equipos
        home_data = team_features[team_features['TEAM_ID'] == pred_home_team_id].iloc[0]
        away_data = team_features[team_features['TEAM_ID'] == pred_away_team_id].iloc[0]
        
        # Preparar features para predicción (simulado - deberías usar las features reales de tu modelo)
        simulation_features = {
            'home_win_pct': home_data.get('HOME_WIN_PCT', 0.5),
            'away_win_pct': away_data.get('AWAY_WIN_PCT', 0.5),
            'home_pts_mean': home_data.get('HOME_PTS_MEAN', 100),
            'away_pts_mean': away_data.get('AWAY_PTS_MEAN', 100),
            'home_advantage': home_data.get('HOME_ADVANTAGE_PTS', 0),
            'home_win_pct_away': home_data.get('AWAY_WIN_PCT', 0.5),
            'away_win_pct_home': away_data.get('HOME_WIN_PCT', 0.5)
        }
        
        # Convertir a array para predicción
        feature_array = np.array(list(simulation_features.values())).reshape(1, -1)
        
        # Realizar predicción basada en el modelo seleccionado
        metric_key = [k for k, v in metric_options.items() if v == selected_metric][0]
        
        if regression_models and metric_key in regression_models:
            model = regression_models[metric_key]
            try:
                prediction = model.predict(feature_array)[0]
                
                # Mostrar resultado de la predicción
                st.subheader("🎯 Resultado de la Predicción")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if metric_key == 'away_weakness':
                        st.metric("Debilidad Visitante Predicha", f"{prediction:.3f}")
                        st.info("Valor más alto indica mayor debilidad del visitante")
                    
                    elif metric_key == 'local_strength':
                        st.metric("Fuerza Local Predicha", f"{prediction:.3f}")
                        st.info("Valor más alto indica mayor ventaja del local")
                    
                    elif metric_key == 'point_differential':
                        st.metric("Diferencia de Puntos Predicha", f"{prediction:.1f}")
                        st.info("Valor positivo favorece al local, negativo al visitante")
                
                with col2:
                    # Mostrar features utilizadas
                    st.subheader("📋 Features Utilizadas")
                    features_df = pd.DataFrame({
                        'Feature': list(simulation_features.keys()),
                        'Valor': list(simulation_features.values())
                    })
                    st.dataframe(features_df, width='stretch')
                    
            except Exception as e:
                st.error(f"Error en la predicción: {e}")
                st.info("""
                **Solución:**
                - Verifica que los modelos de regresión se hayan cargado correctamente
                - Asegúrate de que las features del modelo coincidan con las disponibles
                """)
        else:
            st.warning(f"Modelo para {selected_metric} no disponible")
            st.info("""
            **Para habilitar las predicciones:**
            1. Ejecuta el pipeline completo de Kedro
            2. Asegúrate de que los modelos de regresión se generen correctamente
            3. Verifica los permisos de los archivos .pkl
            """)
        
        # Mostrar análisis comparativo incluso sin modelo
        st.subheader("📊 Análisis Comparativo")
        
        comp_metrics = ['HOME_WIN_PCT', 'AWAY_WIN_PCT', 'HOME_PTS_MEAN', 'AWAY_PTS_MEAN']
        available_metrics = [m for m in comp_metrics if m in home_data and m in away_data]
        
        if available_metrics:
            comparison_data = []
            for metric in available_metrics:
                comparison_data.append({
                    'Métrica': metric,
                    pred_home_team_name: home_data[metric],
                    pred_away_team_name: away_data[metric]
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, width='stretch')

def show_home(data_dict):
    """Página de inicio"""
    st.header("Bienvenido al Sistema de Predicción de Partidos NBA")
    
    # Métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'team_features' in data_dict:
            n_teams = data_dict['team_features']['TEAM_ID'].nunique()
            st.metric("Equipos Analizados", n_teams)
        else:
            st.metric("Equipos Analizados", "N/A")
    
    with col2:
        if 'game_features' in data_dict:
            n_games = len(data_dict['game_features'])
            st.metric("Partidos Analizados", n_games)
        else:
            st.metric("Partidos Analizados", "N/A")
    
    with col3:
        if 'clustering_data' in data_dict:
            n_seasons = data_dict['clustering_data']['SEASON'].nunique()
            st.metric("Temporadas", n_seasons)
        else:
            st.metric("Temporadas", "N/A")
    
    with col4:
        n_models = len(data_dict.get('regression_models', {}))
        st.metric("Modelos Cargados", n_models)
    
    st.markdown("---")
    
    # Enlace rápido a predicciones
    st.subheader("🚀 Comenzar Análisis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Analizar Equipos Visitantes", use_container_width=True):
            st.session_state.selected_page = "Predicciones"
            st.rerun()
    
    with col2:
        if st.button("🎯 Predecir Ganador", use_container_width=True):
            st.session_state.selected_page = "Predicciones" 
            st.rerun()
    
    with col3:
        if st.button("📈 Predecir Métricas", use_container_width=True):
            st.session_state.selected_page = "Predicciones"
            st.rerun()
    
    # Análisis rápido de equipos
    if 'team_features' in data_dict:
        st.subheader("🏆 Top 10 Equipos por Rendimiento")
        
        team_features = data_dict['team_features']
        
        if 'HOME_WIN_PCT' in team_features.columns and 'AWAY_WIN_PCT' in team_features.columns:
            team_performance = team_features.groupby('TEAM_ID').agg({
                'HOME_WIN_PCT': 'mean',
                'AWAY_WIN_PCT': 'mean'
            }).round(3)
            
            # Agregar nombres si están disponibles
            if 'team_names' in data_dict and data_dict['team_names']:
                team_performance['TEAM_NAME'] = team_performance.index.map(data_dict['team_names'])
            
            team_performance['OVERALL_WIN_PCT'] = (team_performance['HOME_WIN_PCT'] + team_performance['AWAY_WIN_PCT']) / 2
            top_teams = team_performance.nlargest(10, 'OVERALL_WIN_PCT')
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'TEAM_NAME' in top_teams.columns:
                    display_df = top_teams[['TEAM_NAME', 'HOME_WIN_PCT', 'AWAY_WIN_PCT', 'OVERALL_WIN_PCT']]
                    display_df.columns = ['Equipo', 'Win% Casa', 'Win% Fuera', 'Win% General']
                else:
                    display_df = top_teams[['HOME_WIN_PCT', 'AWAY_WIN_PCT', 'OVERALL_WIN_PCT']]
                    display_df.columns = ['Win% Casa', 'Win% Fuera', 'Win% General']
                
                st.dataframe(display_df, width='stretch')
            
            with col2:
                x_data = top_teams['TEAM_NAME'] if 'TEAM_NAME' in top_teams.columns else top_teams.index
                fig = px.bar(
                    top_teams.reset_index(),
                    x=x_data,
                    y='OVERALL_WIN_PCT',
                    title='Top 10 Equipos por Porcentaje de Victorias'
                )
                st.plotly_chart(fig, use_container_width=True, config=plotly_config)

def show_team_analysis(data_dict):
    """Análisis detallado de equipos"""
    st.header("📊 Análisis de Equipos NBA")
    
    if 'team_features' not in data_dict:
        st.warning("No se encontraron datos de equipos.")
        return
    
    team_features = data_dict['team_features']
    team_names = data_dict.get('team_names', {})
    
    # Selector de equipo
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if 'TEAM_NAME' in team_features.columns:
            team_options = list(zip(team_features['TEAM_NAME'].unique(), 
                                  team_features['TEAM_ID'].unique()))
            selected_team_name, selected_team_id = st.selectbox(
                "Selecciona un equipo:", 
                team_options,
                format_func=lambda x: x[0]
            )
        else:
            selected_team_id = st.selectbox("Selecciona un equipo:", 
                                           team_features['TEAM_ID'].unique())
            selected_team_name = selected_team_id
        
        # Filtro por temporada si existe
        if 'SEASON' in team_features.columns:
            seasons = team_features['SEASON'].unique()
            selected_season = st.selectbox("Selecciona temporada:", ["Todas"] + list(seasons))
    
    # Datos del equipo seleccionado
    team_data = team_features[team_features['TEAM_ID'] == selected_team_id]
    
    if selected_season != "Todas":
        team_data = team_data[team_data['SEASON'] == selected_season]
    
    if not team_data.empty:
        latest_data = team_data.iloc[-1] if len(team_data) > 1 else team_data.iloc[0]
        
        # Métricas principales
        st.subheader(f"🏀 Rendimiento de {selected_team_name}")
        
        # Crear métricas dinámicas
        metrics_config = [
            ('Win% en Casa', 'HOME_WIN_PCT', '{:.3f}'),
            ('Win% Fuera', 'AWAY_WIN_PCT', '{:.3f}'),
            ('Puntos en Casa', 'HOME_PTS_MEAN', '{:.1f}'),
            ('Puntos Fuera', 'AWAY_PTS_MEAN', '{:.1f}'),
            ('Ventaja Local', 'HOME_ADVANTAGE_PTS', '{:.1f}'),
            ('Diferencia Puntos', 'HOME_PT_DIFF_MEAN', '{:.1f}')
        ]
        
        # Filtrar métricas disponibles
        available_metrics = [(name, col, fmt) for name, col, fmt in metrics_config if col in latest_data]
        
        if available_metrics:
            cols = st.columns(len(available_metrics))
            for idx, (name, col_name, format_str) in enumerate(available_metrics):
                with cols[idx]:
                    st.metric(name, format_str.format(latest_data[col_name]))
        
        # Gráficos de rendimiento
        st.subheader("📈 Tendencias de Rendimiento")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'HOME_WIN_PCT' in team_data.columns and 'AWAY_WIN_PCT' in team_data.columns:
                fig = go.Figure()
                
                # Determinar el eje x
                if 'SEASON' in team_data.columns:
                    x_data = team_data['SEASON']
                    x_label = 'Temporada'
                else:
                    x_data = team_data.index
                    x_label = 'Registro'
                
                fig.add_trace(go.Scatter(x=x_data, y=team_data['HOME_WIN_PCT'], 
                                       name='Win% Casa', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=x_data, y=team_data['AWAY_WIN_PCT'], 
                                       name='Win% Fuera', line=dict(color='red')))
                fig.update_layout(title='Evolución de Porcentaje de Victorias', 
                                xaxis_title=x_label, height=400)
                st.plotly_chart(fig, use_container_width=True, config=plotly_config)
        
        with col2:
            if 'HOME_PTS_MEAN' in team_data.columns and 'AWAY_PTS_MEAN' in team_data.columns:
                fig = go.Figure()
                
                if 'SEASON' in team_data.columns:
                    x_data = team_data['SEASON']
                    x_label = 'Temporada'
                else:
                    x_data = team_data.index
                    x_label = 'Registro'
                
                fig.add_trace(go.Scatter(x=x_data, y=team_data['HOME_PTS_MEAN'], 
                                       name='PTS Casa', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=x_data, y=team_data['AWAY_PTS_MEAN'], 
                                       name='PTS Fuera', line=dict(color='orange')))
                fig.update_layout(title='Evolución de Puntos Promedio', 
                                xaxis_title=x_label, height=400)
                st.plotly_chart(fig, use_container_width=True, config=plotly_config)
        
        # Comparativa con otros equipos
        st.subheader("📊 Comparativa con la Liga")
        
        if 'HOME_WIN_PCT' in team_features.columns:
            league_avg_home = team_features['HOME_WIN_PCT'].mean()
            league_avg_away = team_features['AWAY_WIN_PCT'].mean()
            
            team_home = latest_data['HOME_WIN_PCT']
            team_away = latest_data['AWAY_WIN_PCT']
            
            comp_data = pd.DataFrame({
                'Métrica': ['Win% Casa', 'Win% Fuera'],
                f'{selected_team_name}': [team_home, team_away],
                'Promedio Liga': [league_avg_home, league_avg_away]
            })
            
            fig = px.bar(comp_data, x='Métrica', y=[f'{selected_team_name}', 'Promedio Liga'],
                        title=f'{selected_team_name} vs Promedio de la Liga', barmode='group')
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)

def show_game_analysis(data_dict):
    """Análisis de partidos"""
    st.header("🎯 Análisis de Partidos")
    
    if 'game_features' not in data_dict:
        st.warning("No se encontraron datos de partidos.")
        return
    
    game_features = data_dict['game_features']
    
    st.subheader("Resumen de Partidos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_games = len(game_features)
        st.metric("Total Partidos", total_games)
    
    with col2:
        if 'HOME_TEAM_WINS' in game_features.columns:
            home_wins = game_features['HOME_TEAM_WINS'].sum()
            home_win_pct = (home_wins / total_games) * 100
            st.metric("Victorias Local", f"{home_win_pct:.1f}%")
    
    with col3:
        if 'PTS_home' in game_features.columns:
            avg_points = game_features['PTS_home'].mean()
            st.metric("Puntos Promedio Local", f"{avg_points:.1f}")
    
    with col4:
        if 'PTS_away' in game_features.columns:
            avg_points_away = game_features['PTS_away'].mean()
            st.metric("Puntos Promedio Visitante", f"{avg_points_away:.1f}")
    
    # Distribución de puntos
    st.subheader("Distribución de Puntos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'PTS_home' in game_features.columns:
            fig = px.histogram(game_features, x='PTS_home', 
                             title='Distribución de Puntos - Equipo Local',
                             nbins=30)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
    
    with col2:
        if 'PTS_away' in game_features.columns:
            fig = px.histogram(game_features, x='PTS_away', 
                             title='Distribución de Puntos - Equipo Visitante',
                             nbins=30)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
    
    # Sample de datos
    st.subheader("Muestra de Datos de Partidos")
    st.dataframe(game_features.head(10), width='stretch')

def show_clustering_analysis(data_dict):
    """Análisis de clustering"""
    st.header("🔍 Análisis de Clustering")
    
    if 'clustering_data' not in data_dict:
        st.warning("No se encontraron datos de clustering.")
        return
    
    clustering_data = data_dict['clustering_data']
    
    st.subheader("Datos para Clustering")
    st.write(f"**Dimensiones:** {clustering_data.shape}")
    st.dataframe(clustering_data.head(10), width='stretch')
    
    # Análisis básico de los datos de clustering
    st.subheader("Estadísticas de Clustering")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_teams = clustering_data['TEAM_ID'].nunique()
        st.metric("Equipos Únicos", n_teams)
    
    with col2:
        n_seasons = clustering_data['SEASON'].nunique()
        st.metric("Temporadas", n_seasons)
    
    with col3:
        n_features = len(clustering_data.columns) - 2  # Excluir TEAM_ID y SEASON
        st.metric("Features", n_features)
    
    # Visualización de correlaciones entre features
    st.subheader("Análisis de Features")
    
    # Seleccionar solo columnas numéricas
    numeric_cols = clustering_data.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        # Matriz de correlación
        corr_matrix = clustering_data[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       title='Matriz de Correlación entre Features',
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)
        
        # Scatter plots de features importantes
        st.subheader("Relación entre Features Principales")
        
        important_features = [col for col in numeric_cols if any(keyword in col for keyword in ['WIN', 'PTS', 'PCT'])]
        
        if len(important_features) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(clustering_data, x=important_features[0], y=important_features[1],
                               title=f'{important_features[0]} vs {important_features[1]}',
                               hover_data=['TEAM_ID'])
                st.plotly_chart(fig, use_container_width=True, config=plotly_config)
            
            with col2:
                if len(important_features) >= 4:
                    fig = px.scatter(clustering_data, x=important_features[2], y=important_features[3],
                                   title=f'{important_features[2]} vs {important_features[3]}',
                                   hover_data=['TEAM_ID'])
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

def main():
    # Sidebar
    st.sidebar.title("🏀 Navegación")
    
    # Inicializar session state para navegación
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Inicio"
    
    # Cargar datos
    with st.spinner("Cargando datos NBA..."):
        data_dict = load_data()
    
    # Menú de navegación
    page_options = ["Inicio", "Predicciones", "Análisis de Equipos", "Análisis de Partidos", "Clustering"]
    selected_page = st.sidebar.selectbox("Selecciona una página:", page_options)
    
    # Navegación entre páginas
    if selected_page == "Inicio":
        show_home(data_dict)
    elif selected_page == "Predicciones":
        show_prediction_forms(data_dict)
    elif selected_page == "Análisis de Equipos":
        show_team_analysis(data_dict)
    elif selected_page == "Análisis de Partidos":
        show_game_analysis(data_dict)
    elif selected_page == "Clustering":
        show_clustering_analysis(data_dict)
    
    # Footer informativo
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **🎯 Funcionalidades:**
    - 📊 Análisis equipos visitantes
    - 🎯 Predicción ganador local
    - 📈 Predicción métricas
    - 📋 Análisis comparativos
    """)

if __name__ == "__main__":
    main()