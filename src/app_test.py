import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print("URL:", url)
print("KEY (first 10 chars):", key[:10] if key else None)

try:
    client = create_client(url, key)
    res = client.table("profiles").select("*").limit(1).execute()
    print("✅ Success:", res)
except Exception as e:
    print("❌ Error:", e)
