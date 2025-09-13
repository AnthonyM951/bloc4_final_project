# Gestion des incidents

## Exemple
- Crash d'un worker GPU entraînant l'échec d'un job.

## Procédure
1. Consulter les logs pour identifier la cause.
2. Redémarrer le container affecté.
3. Relancer le job et notifier l'utilisateur.

## Communication
- Notification via Slack et email.
- Objectif : rétablissement du service en < 15 minutes.
