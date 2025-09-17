# Cahier de recettes & tests

## 1. Contexte et objectifs
Ce cahier de recettes couvre les fonctionnalités majeures de la plateforme de génération vidéo IA. Il vise à garantir l'absence d'anomalies fonctionnelles, structurelles et de sécurité, et à détecter rapidement toute régression lors des itérations futures.

## 2. Environnement de test
- **Application** : API Flask + interface HTML minimaliste.
- **Base de données** : PostgreSQL Supabase (tables `users`, `jobs`, `kpi_spark_daily`).
- **File d'attente** : Redis + worker Celery pour la génération vidéo.
- **Jeu de données** : prompts exemples fournis (`prompt_basique`, `prompt_avance`).
- **Rôles testés** : visiteur, utilisateur authentifié, administrateur.

## 3. Scénarios de tests fonctionnels
| ID | Fonctionnalité | Pré-requis | Étapes | Résultat attendu |
|----|----------------|------------|--------|------------------|
| F1 | Inscription utilisateur | Supabase initialisé, application en ligne | 1. Aller sur `/register`. <br>2. Saisir email valide, mot de passe conforme (≥8 caractères). <br>3. Valider. | Création d'un compte, redirection vers `/login`, email inscrit en base.
| F2 | Connexion utilisateur | Compte créé (F1) | 1. Aller sur `/login`. <br>2. Saisir identifiants valides. <br>3. Valider. | Token de session généré, accès au tableau de bord utilisateur.
| F3 | Soumission d'un prompt simple | Utilisateur connecté (F2), worker Celery actif | 1. Accéder à `/jobs/new`. <br>2. Saisir prompt "Un chat jouant du piano". <br>3. Cliquer sur « Générer ». | Job créé en statut `pending`, réponse API `202 Accepted` avec identifiant de job.
| F4 | Suivi de job en cours | Job créé (F3) | 1. Interroger `/jobs/<id>` ou rafraîchir l'UI. | Statut passe successivement de `pending` à `processing`, puis `succeeded` ou `failed`.
| F5 | Téléchargement de la vidéo | Job réussi (F4) | 1. Depuis `/jobs/<id>`, cliquer sur « Télécharger ». | Lien de téléchargement valide, fichier vidéo récupéré (<200 Mo).
| F6 | Historique utilisateur | Jobs existants pour l'utilisateur | 1. Accéder à `/jobs`. | Liste paginée des jobs avec filtres (statut, date). Aucun job d'un autre utilisateur n'est visible.
| F7 | Dashboard administrateur KPIs | Compte admin, pipeline Spark exécuté | 1. Se connecter avec rôle admin. <br>2. Accéder à `/admin/kpis`. | KPIs quotidiens affichés, données alignées avec table `kpi_spark_daily`.
| F8 | Relance d'un job échoué | Job en statut `failed` | 1. Cliquer sur « Relancer ». | Nouveau job créé avec même paramètres, ancien job archivé.
| F9 | Suppression de compte | Utilisateur connecté | 1. Accéder à `/profile`. <br>2. Cliquer sur « Supprimer mon compte ». <br>3. Confirmer. | Compte marqué `deleted`, jobs associés anonymisés, redirection page d'accueil.

## 4. Scénarios de tests structurels
| ID | Cible | Étapes | Résultat attendu |
|----|-------|--------|------------------|
| S1 | API REST | Lancer `pytest tests/test_api.py::test_openapi_contract`. | Le schéma OpenAPI généré correspond au contrat (statuts HTTP, champs).
| S2 | Pipeline Spark KPIs | Exécuter `spark-submit src/pipelines/spark_kpis.py --dry-run`. | Aucun échec, tables temporaires détruites, plan d'exécution conforme.
| S3 | Worker Celery | Lancer `pytest tests/test_worker.py`. | Les tâches utilisent bien les options de réessai (max 3) et la connexion fal.ai est mockée.
| S4 | Intégration Supabase | Lancer `pytest tests/test_supabase_models.py`. | Modèles alignés avec les contraintes (NOT NULL, clés étrangères) de la base.

## 5. Scénarios de tests de sécurité
| ID | Risque | Étapes | Résultat attendu |
|----|--------|--------|------------------|
| SEC1 | Accès API sans authentification | Appeler `POST /api/jobs` sans JWT. | Réponse `401 Unauthorized`, aucun job créé.
| SEC2 | Escalade de privilèges | Utilisateur standard tente d'accéder à `/admin/kpis`. | Réponse `403 Forbidden`, journalisation de l'incident.
| SEC3 | Injection SQL | Soumettre un prompt `'; DROP TABLE jobs; --`. | Saisie stockée via requêtes préparées, job créé normalement, aucune table impactée.
| SEC4 | Fuite de données | Télécharger un job appartenant à un autre utilisateur via URL devinée. | Réponse `404` ou `403`, aucun accès aux fichiers privés.
| SEC5 | Politique de mots de passe | Créer un compte avec mot de passe faible `test123`. | Erreur de validation, message « mot de passe trop faible ».

## 6. Résultats attendus globaux
- Taux de réussite des tests ≥ 95 % sur une exécution complète.
- Temps moyen de génération vidéo < 120 s pour les prompts standards.
- Aucune régression bloquante détectée dans les pipelines CI/CD.
- Journal d'exécution conservé (tests + logs applicatifs) pour audit.

## 7. Gestion des anomalies
- Toute anomalie constatée fait l'objet d'une fiche dans l'outil de suivi (Jira).
- Priorisation : Bloquant > Majeur > Mineur. Correction planifiée avant mise en production.
- Chaque correctif est validé par un rerun complet des scénarios concernés.

## 8. Validation
La recette est prononcée lorsque l'ensemble des scénarios ci-dessus sont exécutés avec succès et que le comité projet (PO, Tech Lead, QA) approuve les rapports d'exécution.
