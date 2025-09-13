# Pipeline CI/CD

1. **Tests unitaires** exécutés avec `pytest`.
2. **Build** de l'image Docker de l'application Flask.
3. **Déploiement** automatique sur un environnement de staging via Docker Compose.
4. Déploiement en production via **Kubernetes** et charts Helm.
