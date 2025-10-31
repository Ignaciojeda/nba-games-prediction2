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
    'nba_classification_models',
    default_args=default_args,
    description='NBA Classification Models - Win Prediction',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['nba', 'classification', 'ml']
) as dag:

    # 1. Data Preparation for Classification
    data_prep_classification = BashOperator(
        task_id='data_prep_classification',
        bash_command='cd /opt/airflow/nba_project && python -m kedro run --pipeline de',
    )

    # 2. Train Classification Models
    train_classification = BashOperator(
        task_id='train_classification_models',
        bash_command='cd /opt/airflow/nba_project && python -m kedro run --pipeline clf',
    )

    # 3. Version Results
    version_classification = BashOperator(
        task_id='version_classification_results',
        bash_command='cd /opt/airflow/nba_project && dvc commit --force && dvc push ',
    )

    # Dependencies
    data_prep_classification >> train_classification >> version_classification