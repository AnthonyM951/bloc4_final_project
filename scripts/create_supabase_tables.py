"""Script de création des tables nécessaires sur la base Supabase.

Ce script utilise psycopg2 pour exécuter des commandes SQL directement
sur la base de données PostgreSQL hébergée par Supabase. Il est attendu
que la variable d'environnement `SUPABASE_DB_URL` contienne la chaîne de
connexion complète vers la base.

Utilisation::

    export SUPABASE_DB_URL="postgresql://..."  # clé service role
    python scripts/create_supabase_tables.py
"""

from __future__ import annotations

import os
import psycopg2

CREATE_PROFILES = """
create table if not exists profiles (
    user_id uuid primary key references auth.users(id),
    username text,
    email text unique not null,
    password_hash text not null,
    created_at timestamp with time zone default now()
);
"""

def main() -> None:
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL non définie")
    conn = psycopg2.connect(db_url)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(CREATE_PROFILES)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
