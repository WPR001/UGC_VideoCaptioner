import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai.errors import ServerError

# === 配置 ===
API_KEY       = ""  # 替换为你的 Key
INPUT_JSONL   = "gemma_e4b_ocr_output.jsonl"
OUTPUT_JSONL  = "gemma_e4b_ocr_output_cleaned.jsonl"
NUM_THREADS   = 5

# 限速参数：Gemini 每分钟最高 1000 请求
INTERVAL = 60.0 / 1000
next_call = time.time()
time_lock = threading.Lock()

# 初始化客户端
client = genai.Client(api_key=API_KEY)

def rate_limit():
    global next_call
    with time_lock:
        now = time.time()
        if next_call > now:
            time.sleep(next_call - now)
        next_call = max(next_call, time.time()) + INTERVAL

def clean_predict_ocr(raw_text: str) -> str:
    """
    把整个 predict_ocr 字符串丢给 Gemini，
    返回它原样的 resp.text（一行输出）。
    """
    prompt = (
        "下面是直接从 OCR 系统得到的原始 predict_ocr 字段内容，可能包含多余描述、注释或格式：\n"
        f"{raw_text}\n\n"
        "请严格提取其中的文本列表，"
        "并以单行 JSON 格式输出，"
        "不要添加其他说明。例如：\n"
        "{\"ocr\": [\"text1\", \"text2\", ...]}"
    )
    for attempt in range(1, 6):
        rate_limit()
        try:
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt]
            )
            return resp.text.strip().replace("\n", "")
        except ServerError as e:
            print(f"[clean] ServerError attempt {attempt}: {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"[clean] Error attempt {attempt}: {e}")
            time.sleep(2 ** attempt)
    return "{\"ocr\": []}"

def process_line(idx: int, line: str) -> (int, str):
    """
    读取一行 JSON，调用 clean_predict_ocr，
    返回 (原始行号, 清洗后的 JSONL 字符串)。
    """
    item = json.loads(line)
    raw_text = item.get("predict_ocr", "")
    cleaned_text = clean_predict_ocr(raw_text)
    item["predict_ocr"] = cleaned_text
    return idx, json.dumps(item, ensure_ascii=False)

def main():
    # 1) 读取所有行
    with open(INPUT_JSONL, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    # 2) 先清空输出文件
    open(OUTPUT_JSONL, 'w', encoding='utf-8').close()

    # 3) 并行处理并实时写入
    with open(OUTPUT_JSONL, 'a', encoding='utf-8') as fout, \
         ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:

        # 提交所有任务
        future_to_idx = {
            executor.submit(process_line, idx, line): idx
            for idx, line in enumerate(lines)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                _, cleaned_line = future.result()
            except Exception as e:
                print(f"[line {idx}] failed: {e}")
                cleaned_line = lines[idx].strip()
            # 立即写入并 flush
            fout.write(cleaned_line + "\n")
            fout.flush()
            print(f"[line {idx}] saved")

    print(f"✅ 清洗完成，结果已写入 {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()