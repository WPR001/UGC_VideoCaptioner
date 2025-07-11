import os
import json
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText


GEMMA_MODEL_ID = "/home/workspace/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, device_map="auto")
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_MODEL_ID, torch_dtype="auto", device_map="auto"
)

# === 数据路径 ===
frames_dir = "/home/workspace/benchmark/frames"  # 每个子目录为一个视频id，包含图像帧
audio_dir = "/home/workspace/benchmark/audio"    # 每个音频文件名为 <video_id>.mp3
output_jsonl = "/home/workspace/benchmark/gemma3n-e2b_results.jsonl"

# === 提示词定义 ===
QUESTION_PROMPT = (
    "You are given a short video with both audio and visual content. "
    "Write a detailed and coherent paragraph that naturally integrates all modalities. "
    "Your description should include: (1) the primary scene and background setting; "
    "(2) key characters or objects and their actions or interactions; "
    "(3) significant audio cues such as voices, background music, sound effects, and their emotional tone; "
    "(4) any on-screen text (OCR) and its role in the video context; and "
    "(5) the overall theme or purpose of the video. "
    "Ensure the output is a fluent and objective paragraph, not a bullet-point list, "
    "and captures the video's content in a human-like, narrative style."
)

# === 获取视频id列表 ===
video_ids = [vid for vid in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, vid))]

with open(output_jsonl, "w", encoding="utf-8") as fout:
    for vid in tqdm(video_ids):
        frame_folder = os.path.join(frames_dir, vid)
        audio_path = os.path.join(audio_dir, f"{vid}.mp3")

        if not os.path.exists(audio_path):
            print(f"[跳过] 音频文件不存在: {audio_path}")
            continue

        # === 收集图像帧 ===
        frame_files = sorted([
            f for f in os.listdir(frame_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if not frame_files:
            print(f"[跳过] 无图像帧: {frame_folder}")
            continue

        # === 构建 multimodal content ===
        content = []
        for file in frame_files:
            frame_path = os.path.join(frame_folder, file)
            content.append({"type": "image", "url": frame_path})
        content.append({"type": "audio", "audio": audio_path})
        content.append({"type": "text", "text": QUESTION_PROMPT})

        messages = [{"role": "user", "content": content}]

        try:
            # === 处理输入 ===
            input_ids = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = input_ids.to(model.device, dtype=model.dtype)

            # === 生成输出 ===
            outputs = model.generate(**input_ids, max_new_tokens=1024)

            # === 解码 + 提取 \nmodel\n 后的内容 ===
            raw_output = processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            if "\nmodel\n" in raw_output:
                model_text = raw_output.split("\nmodel\n", 1)[1].strip()
            else:
                model_text = raw_output.strip()

            # === 写入 JSONL ===
            json.dump({"video_id": vid, "caption": model_text}, fout, ensure_ascii=False)
            fout.write("\n")

        except Exception as e:
            print(f"[错误] 处理 {vid} 时出错: {e}")