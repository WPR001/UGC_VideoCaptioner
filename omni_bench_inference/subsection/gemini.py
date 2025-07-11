import os
import json
import time
import threading
from tqdm import tqdm
from google import genai
from google.genai.errors import ServerError


API_KEY = ""
# API_KEY = ""
INPUT_JSON = ""  # 原始 JSON 路径
VIDEO_FOLDER = ""  # 视频文件所在文件夹
OUTPUT_JSON = ""  # 输出保存路径

MAX_RETRIES = 5
INTERVAL = 60.0 / 1000  
client = genai.Client(api_key=API_KEY)


time_lock = threading.Lock()
next_time = time.time()

def rate_limit():
    global next_time
    with time_lock:
        now = time.time()
        if next_time > now:
            time.sleep(next_time - now)
        next_time = max(next_time, time.time()) + INTERVAL

def safe_upload(video_path, video_id):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            file_ref = client.files.upload(file=video_path)
            while getattr(file_ref, "state", None) == "PROCESSING":
                print(f"[{video_id}] Waiting for processing...")
                time.sleep(5)
                file_ref = client.files.get(name=file_ref.name)
            return file_ref
        except Exception as e:
            print(f"[{video_id}] Upload error (attempt {attempt}): {e}")
            time.sleep(2 ** attempt)
    print(f"[{video_id}] Upload failed after {MAX_RETRIES} attempts.")
    return None

def generate_response(prompt, file_ref, video_id):
    for attempt in range(1, MAX_RETRIES + 1):
        rate_limit()
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[prompt, file_ref]
            )
            if response and response.text:
                return response.text.strip()
            break
        except ServerError as e:
            print(f"[{video_id}] Server error (attempt {attempt}): {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"[{video_id}] Generation error (attempt {attempt}): {e}")
            time.sleep(2 ** attempt)
    return None


def process_all():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = data["question"]
    samples = data["samples"]


    processed_ids = set()
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            for s in existing_data.get("samples", []):
                if "predict" in s and s["predict"] not in [None, ""]:
                    processed_ids.add(s["video_id"])


    output_data = {
        "question": prompt,
        "samples": samples  
    }

    for idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
        video_id = sample["video_id"]


        if video_id in processed_ids:
            print(f"[{video_id}] Already in output. Skipping.")
            continue

        video_path = os.path.join(VIDEO_FOLDER, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            print(f"[{video_id}] Video not found: {video_path}")
            sample["predict"] = None
        else:
            file_ref = safe_upload(video_path, video_id)
            if file_ref is None:
                sample["predict"] = None
            else:
                response = generate_response(prompt, file_ref, video_id)
                sample["predict"] = response if response else None
                try:
                    client.files.delete(name=file_ref.name)
                except Exception as e:
                    print(f"[{video_id}] Failed to delete remote file: {e}")


        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ All done. Results continuously saved to: {OUTPUT_JSON}")



if __name__ == "__main__":
    process_all()