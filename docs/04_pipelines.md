# Pipelines de traitement

## Temps réel
1. L'utilisateur soumet une requête via l'API Flask.
2. Un message est placé dans une file de jobs.
3. Un worker GPU récupère le job et génère la vidéo.
4. Les métadonnées sont sauvegardées dans PostgreSQL et le fichier sur S3.

## Batch avec Airflow
- DAG `video_app_daily` planifié à 02h00 chaque nuit.
- Tâche `aggregate_usage_daily` : agrège les jobs terminés et alimente la table `usage_daily` (jobs_count, gpu_minutes, cost_cents).
- Tâche `prune_intermediate_files` : supprime du stockage les fichiers intermédiaires vieux de plusieurs jours et nettoie la table `files`.

## Calcul distribué (Spark)
- Script `src/pipelines/spark_kpis.py` exécuté via `spark-submit` (local ou cluster).
- Lecture distribuée de la table `jobs` dans PostgreSQL/Supabase via JDBC.
- Agrégations quotidiennes : nombre total de jobs, réussites/échecs, utilisateurs actifs, durées moyenne/médiane et taux de réussite.
- Écriture des résultats dans la table `kpi_spark_daily` (ou un bucket Parquet S3) consommée par le tableau de bord admin `/admin/kpis` et mise à jour lors de chaque exécution.
- Variables d'environnement : `SUPABASE_DB_*` pour la connexion, `SPARK_KPI_OUTPUT_*` pour choisir la destination, `SPARK_JARS_PACKAGES` pour ajouter le driver Postgres.
