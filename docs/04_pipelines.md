# Pipelines de traitement

## Temps réel
1. L'utilisateur soumet une requête via l'API Flask.
2. Un message est placé dans une file de jobs.
3. Un worker GPU récupère le job et génère la vidéo.
4. Les métadonnées sont sauvegardées dans PostgreSQL et le fichier sur S3.

## Batch avec Airflow
- Tâche nocturne d'archivage et de nettoyage des jobs terminés.
- Pipeline Spark pour analyser les logs : durée moyenne, erreurs, coûts GPU.
