from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


def archive_jobs():
    # Placeholder pour l'archivage et le nettoyage
    print("Archivage des jobs terminés")


def analyze_logs():
    # Placeholder pour l'analyse des logs avec Spark
    print("Analyse des logs de génération")


dag = DAG(
    dag_id="archive_and_analyze",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",
    catchup=False,
)

with dag:
    archive = PythonOperator(task_id="archive_jobs", python_callable=archive_jobs)
    analyze = PythonOperator(task_id="analyze_logs", python_callable=analyze_logs)
    archive >> analyze
