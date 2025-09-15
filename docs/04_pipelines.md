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
