from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator

default_args = {
    'owner': 'nba-ml-team',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'nba_regression_models',
    default_args=default_args,
    description='NBA Regression Models - Point Differential',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['nba', 'regression', 'ml']
) as dag:

    # 1. Data Preparation for Regression (SIN parámetros o con formato correcto)
    data_prep_regression = BashOperator(
        task_id='data_prep_regression',
        # OPCIÓN A: Sin parámetros específicos
        bash_command='cd /opt/airflow/nba_project && python -m kedro run --pipeline data_processing',
        
        # OPCIÓN B: Con parámetros en formato correcto (si tu pipeline los soporta)
        # bash_command='cd /opt/airflow/nba_project && python -m kedro run --pipeline data_processing --params "model_type=regression"',
    )

    # 2. Train Regression Models
    train_regression = BashOperator(
        task_id='train_regression_models',
        bash_command='cd /opt/airflow/nba_project && python -m kedro run --pipeline regression_pipeline',
    )

    # 3. Version Results
    version_regression = BashOperator(
        task_id='version_regression_results',
        bash_command='cd /opt/airflow/nba_project && dvc commit && dvc push',
    )

    # Dependencies
    data_prep_regression >> train_regression >> version_regression