# Cahier de recettes & tests

## Tests fonctionnels
- Soumission d'un prompt simple -> réception d'une vidéo.
- Erreur attendue si authentification manquante.

## Tests de charge
- 100 requêtes simultanées déclenchent le scaling automatique des workers GPU.

## Tests de sécurité
- Authentification via JWT obligatoire pour l'API.

## Résultats attendus
- Taux de succès > 95%.
- Temps moyen de génération < 2 minutes.
