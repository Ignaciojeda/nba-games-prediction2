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
    'nba_complete_ml_pipeline',
    default_args=default_args,
    description='NBA Complete ML Pipeline - Classification + Regression',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['nba', 'ml', 'complete']
) as dag:

    # 1. Data Processing
    data_processing = BashOperator(
        task_id='data_processing',
        bash_command='cd /opt/airflow/nba_project && python -m kedro run --pipeline data_processing',
    )

    # 2. Classification Models (Paralelo)
    classification_models = BashOperator(
        task_id='classification_models',
        bash_command='cd /opt/airflow/nba_project && python -m kedro run --pipeline classification_pipeline',
    )

    # 3. Regression Models (Paralelo)
    regression_models = BashOperator(
        task_id='regression_models',
        bash_command='cd /opt/airflow/nba_project && python -m kedro run --pipeline regression_pipeline',
    )

    # 4. DVC Versioning
    dvc_versioning = BashOperator(
        task_id='dvc_versioning',
        bash_command='cd /opt/airflow/nba_project && dvc repro',
    )

    # Dependencies
    data_processing >> [classification_models, regression_models] >> dvc_versioning