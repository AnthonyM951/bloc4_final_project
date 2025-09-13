# Schéma de données

```
+---------+        +-------+        +--------+
|  users  | 1    n | jobs  | 1    1 | videos |
+---------+        +-------+        +--------+
| id      |<-----> | user_id       | job_id |
| name    |        | prompt        | path   |
| email   |        | audio_ref     | status |
| quota   |        | status        | created_at |
| created_at |     | created_at    |         |
+---------+        +-------+        +--------+
```

Accès via API REST Flask et interface web.
