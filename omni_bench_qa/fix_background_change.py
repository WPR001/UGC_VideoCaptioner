import os
import json
import time
import threading
from google import genai
from google.genai.errors import ServerError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# === 配置 ===
API_KEY = "YOUR_API_KEY_HERE"  # 记得替换为你自己的 API Key
INPUT_JSON = "/path/to/your/input.json"
OUTPUT_JSON = "/path/to/your/output_rewritten.json"
MAX_RETRIES = 5
NUM_WORKERS = 5
INTERVAL = 60.0 / 1000  # 每分钟 1000 个请求
MODEL = "gemini-2.5-pro"

client = genai.Client(api_key=API_KEY)

counter_lock = threading.Lock()
time_lock = threading.Lock()
total_requests = 0
next_time = time.time()


def rate_limit():
    global next_time
    with time_lock:
        now = time.time()
        if next_time > now:
            time.sleep(next_time - now)
            now = next_time
        next_time = now + INTERVAL


def safe_upload(file_path, video_id):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            myfile = client.files.upload(file=file_path)
            while getattr(myfile, "state", None) == "PROCESSING":
                print(f"[{video_id}] Waiting for video to process.")
                time.sleep(5)
                myfile = client.files.get(name=myfile.name)
            return myfile
        except Exception as e:
            print(f"[{video_id}] Upload error (attempt {attempt}): {e}")
            time.sleep(2 ** attempt)
    return None


def generate_description(video_file, video_id, original_answer):
    prompt = (
        "You are given a short video and a rough description of background changes during it. "
        "Please rewrite the background change description into a more detailed, fluent paragraph. "
        "Make it human-like, precise, and specific about transitions in background settings. "
        f"The rough answer is: \"{original_answer}\""
    )

    file_ref = safe_upload(video_file, video_id)
    if not file_ref:
        return None

    contents = [prompt, file_ref]

    global total_requests
    for attempt in range(1, MAX_RETRIES + 1):
        with counter_lock:
            rate_limit()
            total_requests += 1
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents
            )
            if response and response.text:
                return response.text.strip()
        except ServerError as e:
            print(f"[{video_id}] ServerError: {e} (attempt {attempt})")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"[{video_id}] Error: {e} (attempt {attempt})")
            time.sleep(2 ** attempt)

    return None


def process_sample(sample):
    video_id = sample["video_id"]
    video_path = sample["path"]
    original_answer = sample.get("answer", "")

    if not os.path.exists(video_path):
        print(f"[{video_id}] Video file not found: {video_path}")
        return None

    rewritten = generate_description(video_path, video_id, original_answer)
    if not rewritten:
        return None

    # 删除上传的视频文件
    try:
        client.files.delete(name=video_id)
    except Exception:
        pass

    return {
        "video_id": video_id,
        "path": video_path,
        "answer": rewritten
    }


def main():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    question = data["question"]
    samples = data["samples"]
    rewritten_samples = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_sample, samples), total=len(samples)))

    # 过滤失败的样本
    rewritten_samples = [r for r in results if r is not None]

    output_data = {
        "question": question,
        "valid_count": len(rewritten_samples),
        "samples": rewritten_samples
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved rewritten results to {OUTPUT_JSON}")


if __name__ == '__main__':
    main()