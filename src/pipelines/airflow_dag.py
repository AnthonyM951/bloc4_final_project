from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator
import os
import psycopg2
from supabase import create_client

# --- Config Supabase Storage pour le nettoyage ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BUCKET_INTERMEDIATE = os.getenv("BUCKET_INTERMEDIATE", "uploads")
DAYS_TO_KEEP_INTERMEDIATE = int(os.getenv("DAYS_TO_KEEP_INTERMEDIATE", "3"))


def prune_intermediate_files(**_):
    """Supprime du Storage les fichiers 'intermediate' vieux de N jours et purge la table files correspondante."""
    # Connexion SQL directe (pratique si pas de PostgresHook)
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_DB_HOST"),
        dbname=os.getenv("SUPABASE_DB_NAME", "postgres"),
        user=os.getenv("SUPABASE_DB_USER", "postgres"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        sslmode="require",
    )
    supa = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    with conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, bucket, path
            FROM files
            WHERE kind = 'intermediate'
              AND created_at < now() - interval '%s day'
            """,
            (DAYS_TO_KEEP_INTERMEDIATE,),
        )
        rows = cur.fetchall()

        for fid, bucket, path in rows:
            try:
                supa.storage.from_(bucket).remove([path])  # ignore si déjà manquant
            except Exception:
                pass
            cur.execute("DELETE FROM files WHERE id = %s", (fid,))


# --- SQL d’agrégation KPI jour ---
UPSERT_USAGE_DAILY = """
insert into usage_daily(day, user_id, jobs_count, gpu_minutes, cost_cents)
select
  date_trunc('day', coalesce(finished_at, submitted_at))::date as day,
  user_id,
  count(*) filter (where status = 'succeeded') as jobs_count,
  coalesce(sum(extract(epoch from (finished_at - started_at)) / 60.0), 0) as gpu_minutes,
  coalesce(sum(cost_cents), 0) as cost_cents
from jobs
group by 1,2
on conflict (day, user_id)
do update set
  jobs_count = excluded.jobs_count,
  gpu_minutes = excluded.gpu_minutes,
  cost_cents = excluded.cost_cents;
"""

default_args = {
    "owner": "data-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="video_app_daily",
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 2 * * *",  # tous les jours à 02:00
    catchup=False,
    default_args=default_args,
    tags=["video-app", "batch"],
) as dag:

    start = EmptyOperator(task_id="start")

    aggregate_usage = PostgresOperator(
        task_id="aggregate_usage_daily",
        postgres_conn_id="supabase_db",
        sql=UPSERT_USAGE_DAILY,
    )

    prune_files = PythonOperator(
        task_id="prune_intermediate_files",
        python_callable=prune_intermediate_files,
    )

    finish = EmptyOperator(task_id="finish")

    start >> aggregate_usage >> prune_files >> finish

