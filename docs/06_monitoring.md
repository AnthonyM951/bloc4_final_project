# Supervision & alertes

- Collecte de logs via **Loki** et visualisation avec **Grafana**.
- Métriques : nombre de jobs, temps de génération, consommation GPU.
- Alertes si taux d'échec > 5% ou indisponibilité d'un worker GPU.
- Orchestration supervisée dans **Airflow UI** : historique des exécutions du DAG, accès aux logs et statut des tâches.
