import os
import time
import requests
import fal_client

# config
FAL_KEY = os.getenv("FAL_KEY")


MODEL = "fal-ai/veo3/fast"
REQUEST_ID = "2e31a061-3fd0-4c0e-9938-cdf48b50d750"

headers = {
    "Authorization": f"Key {FAL_KEY}",
    "Content-Type": "application/json",
}

def poll_result(request_id: str, poll_interval: float = 5.0, max_wait: float = 600.0):
    start = time.time()
    attempt = 0
    while True:
        attempt += 1
        print(f"[Attempt {attempt}] Trying fal_client.result(...)")
        try:
            result = fal_client.result(MODEL, request_id)
        except Exception as e:
            print("  → fal_client.result threw:", repr(e))
        else:
            print("  → result:", result)
            # try extract video URL
            video_url = None
            if isinstance(result, dict):
                for field in ("payload", "response", "data"):
                    payload = result.get(field)
                    if isinstance(payload, dict):
                        video = payload.get("video")
                        if isinstance(video, dict):
                            video_url = video.get("url")
                            if video_url:
                                print("🎉 Video URL found via fal_client.result:", video_url)
                                return video_url
            print("  → no video URL yet from result()")

        # fallback: try queue GET /requests/{request_id} (full payload)
        url = f"https://queue.fal.run/{MODEL}/requests/{request_id}"
        print(f"[Attempt {attempt}] Trying direct GET on {url}")
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            print("  → Status code:", resp.status_code)
            if resp.status_code == 200:
                full = resp.json()
                print("  → Direct GET result:", full)
                video_section = None
                # Try extract video url
                if isinstance(full, dict):
                    for fld in ("payload", "response", "data"):
                        pd = full.get(fld)
                        if isinstance(pd, dict):
                            vid = pd.get("video")
                            if isinstance(vid, dict):
                                video_section = vid.get("url")
                                if video_section:
                                    print("🎉 Video URL via direct GET:", video_section)
                                    return video_section
                print("  → direct GET has no video URL yet")
            else:
                print("  → direct GET returned non-200:", resp.text[:200])
        except Exception as ex2:
            print("  → direct GET threw:", repr(ex2))

        elapsed = time.time() - start
        if elapsed > max_wait:
            print("⏱ Timeout exceeded, giving up.")
            return None

        time.sleep(poll_interval)

if __name__ == "__main__":
    url = poll_result(REQUEST_ID, poll_interval=5, max_wait=600)
    print("Final video URL:", url)
