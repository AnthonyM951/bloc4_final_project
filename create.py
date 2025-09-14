import psycopg2

conn = psycopg2.connect(
    host="db.cryetaumceiljumacrww.supabase.co",
    dbname="postgres",
    user="postgres",
    password="Davidson@2308,,,",
    port=5432,
    sslmode="require"
)
cur = conn.cursor()
cur.execute("""
    create table if not exists my_items (
        id serial primary key,
        name text not null,
        created_at timestamp default now()
    );
""")
conn.commit()
