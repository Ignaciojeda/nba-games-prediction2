# Crear el archivo del DAG en airflow\dags\
@'
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator

default_args = {
    'owner': 'nba-ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'nba_ml_pipeline',
    default_args=default_args,
    description='NBA Machine Learning Pipeline with Kedro and DVC',
    schedule_interval=timedelta(days=1),
    catchup=False
) as dag:

    # 1. Data Processing
    data_processing = BashOperator(
        task_id='data_processing',
        bash_command='cd /opt/airflow/nba_project && kedro run --pipeline=data_engineering',
    )

    # 2. Feature Engineering
    feature_engineering = BashOperator(
        task_id='feature_engineering',
        bash_command='cd /opt/airflow/nba_project && kedro run --pipeline=feature_engineering',
    )

    # 3. Classification Pipeline
    classification = BashOperator(
        task_id='classification_pipeline',
        bash_command='cd /opt/airflow/nba_project && kedro run --pipeline=classification',
    )

    # 4. Regression Pipeline
    regression = BashOperator(
        task_id='regression_pipeline',
        bash_command='cd /opt/airflow/nba_project && kedro run --pipeline=regression',
    )

    # 5. DVC Versioning
    dvc_versioning = BashOperator(
        task_id='dvc_versioning',
        bash_command='cd /opt/airflow/nba_project && dvc repro',
    )

    # Dependencies
    data_processing >> feature_engineering
    feature_engineering >> [classification, regression]
    [classification, regression] >> dvc_versioning
'@ | Out-File -FilePath "airflow\dags\nba_ml_pipeline.py" -Encoding UTF8