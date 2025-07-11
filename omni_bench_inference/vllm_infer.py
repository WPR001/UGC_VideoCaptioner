# pip install vllm
# pip install transformers==4.52.3


import os
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser

VIDEO_DIR = "/workspace/benchmark/video"
OUTPUT_JSONL = "/workspace/benchmark/omni_vllm_sft_result_same_parameter.jsonl"
USE_AUDIO_IN_VIDEO = True
MAX_RETRY = 3

# Ensure output file exists
def ensure_output_file(path: str):
    if not os.path.exists(path):
        open(path, "w", encoding="utf-8").close()

# Load processed video IDs to skip

def load_processed_ids(jsonl_path: str) -> set[str]:
    processed = set()
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                data = json.loads(line)
                vid = data.get("video_id")
                if vid:
                    processed.add(vid)
            except json.JSONDecodeError:
                continue
    return processed

# Regex to verify level tag at end of caption 
# 没有level
LEVEL_PATTERN = re.compile(r"<level>[A-F]</level>\s*$")

PROMPT_TEMPLATE = (
    f"<|im_start|>system\n" +
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Please make sure that the content within the answer is long and detailed enough." +
    "<|im_end|>\n"
    "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
    "You are given a short video with both audio and visual content. Write a detailed and coherent paragraph that naturally integrates all modalities. "
    "Your description should include: (1) the primary scene and background setting; (2) key characters or objects and their actions or interactions; "
    "(3) significant audio cues such as voices, background music, sound effects, and their emotional tone; "
    "(4) any on-screen text (OCR) and its role in the video context; and (5) the overall theme or purpose of the video. "
    "Ensure the output is a fluent and objective paragraph, not a bullet-point list, and captures the video's content in a human-like, narrative style. <|im_end|>\n"
    "<|im_start|>assistant\n"
)


def process_video_folder(model_name: str, seed: int = None):
    ensure_output_file(OUTPUT_JSONL)
    video_files = sorted(f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".mp4"))
    processed_ids = load_processed_ids(OUTPUT_JSONL)

    llm = LLM(
        model=model_name,
        max_model_len=20000,
        max_num_seqs=5,
        limit_mm_per_prompt={"video": 1, "audio": 1},
        seed=seed,
    )
    sampling_params = SamplingParams(temperature=0.2, max_tokens=1024)

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as fout:
        for fname in tqdm(video_files, desc="Processing videos"):
            video_id = os.path.splitext(fname)[0]
            if video_id in processed_ids:
                print(f"[Skip] {fname} already processed, skipping.")
                continue

            fpath = os.path.join(VIDEO_DIR, fname)
            valid_caption = None
            try:
                video_asset = VideoAsset(path=fpath, num_frames=32)
                audio = video_asset.get_audio(sampling_rate=16000)

                inputs = {
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"video": video_asset.np_ndarrays, "audio": audio},
                    "mm_processor_kwargs": {"use_audio_in_video": USE_AUDIO_IN_VIDEO},
                }

                for attempt in range(MAX_RETRY):
                    outputs = llm.generate(inputs, sampling_params=sampling_params)
                    text = outputs[0].outputs[0].text.strip()
                    if text and LEVEL_PATTERN.search(text):
                        valid_caption = text
                        break
                    else:
                        print(f"[Retry] Attempt {attempt+1} for {fname} did not end with level tag, retrying...")

                if not valid_caption:
                    print(f"[Warning] {fname} failed to get valid level tag after {MAX_RETRY} attempts, skipping.")
                    continue

                fout.write(json.dumps({"video_id": video_id, "caption": valid_caption}, ensure_ascii=False) + "\n")
                fout.flush()
                processed_ids.add(video_id)

            except Exception as e:
                print(f"[Error] Failed to process {fname}: {e}")
                continue

    print(f"✅ Done! Processed videos with skipping and level validation. Output written to {OUTPUT_JSONL}")


def parse_args():
    parser = FlexibleArgumentParser(description="Batch inference for a folder of videos using Qwen2.5-Omni + vLLM.")
    parser.add_argument("--model-name", type=str, default="/workspace/output_model/tiktok_caption/omni_sft_20k_level/v1-20250701-150049/checkpoint-2404-merged", help="Model path or name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video_folder(args.model_name, args.seed)
