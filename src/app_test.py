import os
import json
import requests

FAL_KEY = os.getenv("FAL_KEY", "3e9ddf21-a57e-4b69-9eb9-2d9d336acf92:0296b68b75feab14420a58c753385b05")
FAL_QUEUE_BASE = os.getenv("FAL_QUEUE_BASE", "https://queue.fal.run")
MODEL = os.getenv("MODEL_DEFAULT", "fal-ai/infinitalk/single-text")
WEBHOOK_URL = os.getenv("FAL_WEBHOOK_URL", "https://example.com/webhooks/fal")

fal_input = {
    "prompt": "un professeur donne un cours en souriant",
    "text_input": "Tyrannosaurus rex ( rex meaning 'king' in Latin ), often shortened to T. rex or colloquially t-rex , is one of the best represented theropods. It lived throughout what is now western North America , on what was then an island continent known as Laramidia . Tyrannosaurus had a much wider range than other tyrannosaurids . Fossils are found in a variety of geological formations dating",
    "image_url": "https://as1.ftcdn.net/jpg/01/90/54/30/1000_F_190543073_bz6SjkIIx6tEWOquw9oPECIzrPfJtwhL.jpg",
    "voice": "Brian",
    "num_frames": 145,
    "resolution": "480p",
    "seed": 42,
    "acceleration": "high",
    "webhook_url": WEBHOOK_URL,
}

headers = {
    "Authorization": f"Key {FAL_KEY}",
    "Content-Type": "application/json",
}

payload = fal_input

url = f"{FAL_QUEUE_BASE}/{MODEL}"

print("➡️ Submitting job to:", url)
resp = requests.post(url, headers=headers, data=json.dumps(payload))

print("🔄 Status code:", resp.status_code)
try:
    print("📦 Response:", resp.json())
except Exception:
    print("📦 Raw response:", resp.text)
