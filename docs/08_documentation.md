# Documentation technique & fonctionnelle

## 1. Vue d'ensemble
La plateforme génère des vidéos à partir de prompts texte ou audio. Elle repose sur trois
composants principaux :
- **API Flask** exposant les endpoints REST et pilotant la file de jobs.
- **Worker Celery** connecté à un GPU pour orchestrer les modèles de génération.
- **Interface web** (templates Jinja + JS léger) permettant aux utilisateurs de soumettre et
  suivre leurs créations.

## 2. Architecture applicative
```
Navigateur ──> Flask API ──> PostgreSQL (Supabase)
        │           │
        │           └──> Redis (queue Celery) ──> Worker GPU ──> Stockage objets Supabase
        └───────────────────────────────────────────────────────────────────────────────┘
```
- Les jobs sont persistés dans `jobs` et leurs sorties sont stockées dans le bucket
  `jobs-outputs`.
- Les métriques (KPIs quotidiens, durées) sont agrégées via un pipeline Spark alimentant la
  table `kpi_spark_daily`.

## 3. Modules backend
| Module | Rôle | Points clés |
|--------|------|-------------|
| `src/app.py` | Bootstrap Flask, Blueprints `auth`, `jobs`, `admin`. | Charge la config depuis `.env`, enregistre les extensions (SQLAlchemy, JWT). |
| `src/services/jobs.py` | Orchestration des jobs. | Validation des prompts, création des tâches Celery, suivi des statuts. |
| `src/services/storage.py` | Gestion Supabase Storage. | Génère des URLs signées, gère la rétention et le nettoyage. |
| `src/services/notifications.py` | Envoi d'emails SendGrid. | Templates `job_created`, `job_completed`. |
| `src/models` | SQLAlchemy models. | Tables `users`, `jobs`, `job_events`, `kpi_spark_daily`. |

### Endpoints principaux
- `POST /api/register`, `POST /api/login` : gestion des comptes et tokens JWT.
- `POST /api/jobs` : création d'un job (payload `prompt`, `options`).
- `GET /api/jobs/<id>` : statut détaillé d'un job.
- `GET /api/jobs` : pagination et filtres par statut/date.
- `POST /api/jobs/<id>/retry` : relance d'un job échoué (admin).
- `GET /api/kpis/daily` : KPIs agrégés (admin).

Les tests d'intégration se trouvent dans `tests/test_api.py` et doivent rester verts avant
chaque mise en production.

## 4. Worker de génération
- Entrée : message Celery `{job_id, prompt, options}`.
- Étapes :
  1. Téléchargement des modèles depuis le cache local (`/models`).
  2. Génération de la vidéo via `fal_client.generate_video`.
  3. Upload du résultat dans Supabase Storage et mise à jour du statut (`processing` →
     `succeeded`/`failed`).
- Retrys automatiques : 3 tentatives, backoff exponentiel (5s, 15s, 45s).
- Logs : envoyés vers stdout + Loki (via promtail). Les erreurs critiques déclenchent une
  alerte PagerDuty `worker.failure`.

## 5. Pipelines & analytics
- Script `src/pipelines/spark_kpis.py` exécuté chaque nuit (Airflow) :
  - Lecture des jobs de la veille.
  - Agrégation des durées moyennes, taux de succès, consommation de crédits.
  - Écriture dans `kpi_spark_daily`.
- KPI front (`/admin/kpis`) lit directement cette table via l'API.

## 6. Configuration & secrets
| Variable | Description | Exemple |
|----------|-------------|---------|
| `FLASK_ENV` | Environnement (`development`, `production`, `qa`). | `qa` |
| `DATABASE_URL` | Connexion Supabase PostgreSQL. | `postgresql://...` |
| `REDIS_URL` | File d'attente Celery. | `redis://redis:6379/0` |
| `SUPABASE_BUCKET` | Nom du bucket de stockage. | `jobs-outputs` |
| `SENDGRID_API_KEY` | Envoi d'emails. | `SG.xxxxxx` |
| `FAL_API_KEY` | Accès au moteur IA. | `fal-xxxxxxxx` |

Les secrets sont gérés par Vault en production, injectés sous forme de variables d'environnement.
En local, copier `env.example` vers `.env` puis ajuster les valeurs.

## 7. Déploiement
### Docker Compose (local / QA)
1. `docker compose up --build` lance les services `api`, `worker`, `redis`, `db`.
2. Les migrations sont appliquées automatiquement via `flask db upgrade`.
3. Les volumes `./data` conservent les artefacts générés pour inspection.

### Kubernetes (production)
- Namespace `video-gen`.
- Déploiements : `api-deployment` (3 pods), `worker-gpu` (1 pod GPU A10), `redis` (Helm chart).
- Auto-scaling horizontal sur l'API (HPA 30% CPU, 50% RAM).
- Jobs Cron `spark-kpi-cronjob` pour lancer Spark sur un cluster dédié.

## 8. Observabilité & supervision
- **Logs** : centralisés dans Loki, visualisation Grafana (`Dashboard: Video Gen - Logs`).
- **Metrics** : Prometheus scrape `/metrics` (latence API, jobs en attente, durée moyenne).
- **Alertes** :
  - `APIHighErrorRate` (>5% 5xx sur 5 min) → Slack `#alerts-backend`.
  - `WorkerQueueBacklog` (>20 jobs pending 10 min) → PagerDuty.
  - `StorageUsageThreshold` (>80% du bucket) → email plateforme.

## 9. Sécurité & conformité
- Authentification JWT avec refresh tokens 24h.
- Rôles : `user`, `admin`, `support` (lecture seule).
- Audit trail via table `job_events` et webhooks Grafana Loki.
- Sauvegardes quotidiennes Supabase (rétention 30 jours).
- Politique RGPD : suppression/anonymisation sur demande (`DELETE /api/users/me`).

## 10. Guide utilisateur
1. Se connecter à l'interface web.
2. Soumettre un prompt texte ou un fichier audio.
3. Choisir les options (résolution, voix off, durée).
4. Suivre l'avancement depuis le tableau de bord.
5. Télécharger la vidéo générée ou relancer si nécessaire.

## 11. Guide administrateur
- Ajouter un worker GPU : déployer un nouveau pod `worker-gpu` via Helm (`values.gpu=true`).
- Relancer un pod défaillant : `kubectl rollout restart deployment/api-deployment`.
- Consulter les logs : Grafana > Dashboard « Worker GPU ».
- Purger les jobs obsolètes : script `scripts/purge_jobs.py --older-than 30`.

## 12. Installation locale
1. Cloner le dépôt et créer un environnement virtuel Python 3.11.
2. Installer les dépendances : `pip install -r requirements.txt`.
3. Copier `env.example` → `.env` et renseigner les clés (voir §6).
4. Démarrer la stack :
   ```bash
   docker compose up redis db
   flask db upgrade
   flask run  # interface API
   celery -A worker.celery_app worker --loglevel=info
   npm install && npm run dev  # si frontend standalone
   ```
5. Lancer les tests : `pytest` (backend), `npm run test` (frontend), `black --check src` (lint).

## 13. Troubleshooting rapide
| Symptôme | Diagnostic | Action |
|----------|------------|--------|
| Jobs bloqués en `pending` | Worker Celery indisponible. | Vérifier `celery -A worker inspect ping`, relancer le pod si besoin. |
| Erreur 500 sur `/api/jobs` | Variables d'environnement manquantes. | Contrôler `flask config`, recharger `.env`. |
| Vidéo non téléchargeable | URL expirée ou fichier manquant. | Régénérer via `scripts/reissue_signed_url.py`, vérifier Supabase. |
| KPIs vides | Cron Spark en erreur. | Consulter logs Airflow, relancer la tâche `spark_kpi_daily`. |

## 14. Références
- Schémas de données : `docs/03_schema_donnees.md`.
- Pipelines CI/CD : `docs/05_cicd.md`.
- Runbooks incidents : `docs/10_incidents.md`.
