# Analyse du projet

## Besoins métiers
- Génération de vidéos à partir de prompts ou d'audio.
- Service web exposé via Flask.

## Volumes de données
- Fichiers vidéo et audio stockés sur S3/MinIO.
- Journaux et métadonnées en base PostgreSQL.

## Environnement existant
- Framework Flask.
- API de génération IA (Stable Diffusion / ComfyUI).
- Accélération GPU dans le cloud.

## Contraintes
- Coût d'utilisation des GPU.
- Scalabilité et délais de rendu.
- Conformité RGPD pour les données utilisateurs.

## État des lieux
- Infrastructure conteneurisée via Docker.
- Déploiement continu avec GitHub Actions.
- Une VM de développement et possibilité d'orchestration Kubernetes.
