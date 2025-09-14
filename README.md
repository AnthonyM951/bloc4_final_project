# Bloc 4 - Projet génération vidéo IA

Ce dépôt contient une application **Flask** minimale pour la génération de vidéos par IA ainsi que la documentation exigée pour la validation du Bloc 4.

## Structure
- `docs/` : documents d'analyse, architecture, CI/CD, exploitation, etc.
- `src/` : code source de l'application, modèles de données, pipelines et tests.
- `requirements.txt` : dépendances Python minimales.

## Exécution
```bash
pip install -r requirements.txt
pytest
flask run
```

### Worker Celery

Un worker Celery gère la génération vidéo via [fal.ai](https://fal.ai).
Pour l'activer, installez Redis ou configurez une URL de broker, puis
lancez le worker :

```bash
export FAL_KEY="votre_cle_api"
celery -A worker.celery worker --loglevel=info
```

## Intégration Supabase

L'application peut s'appuyer sur [Supabase](https://supabase.com) pour la
gestion des utilisateurs. Pour l'activer :

1. Créez un projet Supabase et récupérez l'URL du projet, la clé anonyme
   et l'URL de connexion à la base de données.
2. Exportez les variables d'environnement nécessaires :

   ```bash
   export SUPABASE_URL="https://xyz.supabase.co"
   export SUPABASE_ANON_KEY="votre_clé_anon"
   export SUPABASE_DB_URL="postgresql://..."  # clé service role
   ```

3. Initialisez la table `profiles` utilisée par le formulaire
   d'inscription :

   ```bash
   python scripts/create_supabase_tables.py
   ```

4. Lancez l'application (`flask run`) puis rendez-vous sur `/register`
   pour créer un compte qui sera stocké dans Supabase.
