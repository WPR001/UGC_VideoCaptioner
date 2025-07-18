
# UGC-VideoCaptioner: An Omni UGC Video Detail Caption Model and New Benchmarks

**UGC-VideoCaptioner** is a 3B-parameter captioning model distilled from Gemini-2.5 Flash, specifically designed for detailed, omnimodal captioning of short-form, user-generated videos (UGC). It addresses the crucial role of audio in conjunction with visual content, which is often overlooked in existing video captioning models.

<p align="left">
  <a href="https://huggingface.co/papers/2507.11336" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace_Paper-2507.11336%F0%9F%93%96-blue">
  </a>
  <a href="http://arxiv.org/abs/2507.11336" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2507.11336%F0%9F%93%96-bron">
  </a>
  <a href="https://memories.ai/" target="_blank">
    <img src="https://img.shields.io/badge/Project_Page-%F0%9F%8C%90-green">
  </a>
  <a href="https://huggingface.co/collections/openinterx/ugc-videocap-6845e290580112a1834737c4" target='_blank'>
    <img src="https://img.shields.io/badge/Huggingface%20Models-🤗-blue">
  </a>
</p>

## Abstract

Real-world user-generated videos, especially on platforms like TikTok, often feature rich and intertwined audio visual content. However, existing video captioning benchmarks and models remain predominantly visual centric, overlooking the crucial role of audio in conveying scene dynamics, speaker intent, and narrative context. This lack of omni datasets and lightweight, capable models hampers progress in fine grained, multimodal video understanding. To address these challenges, we introduce UGC-VideoCap, a new benchmark and model framework specifically designed for detailed omnimodal captioning of short form user-generated videos. Unlike prior datasets, UGC-VideoCap emphasizes balanced integration of audio and visual modalities, featuring 1000 TikTok videos annotated through a structured three stage human-in-the-loop pipeline covering audio only, visual only, and joint audio visual semantics. The benchmark also includes 4000 carefully crafted QA pairs probing both unimodal and cross modal understanding. Alongside the dataset, we propose UGC-VideoCaptioner(3B), a 3B parameter captioning model distilled from Gemini 2.5 Flash. Using a novel two-stage training strategy supervised fine tuning followed by Group Relative Policy Optimization (GRPO), our approach enables efficient adaptation from limited data while maintaining competitive performance. Together, our benchmark and model offer a high-quality foundation and a data-efficient solution for advancing omnimodal video captioning in unconstrained real-world UGC settings.

<p align="center">
    <img src="tiktok_qa_sample.png" alt="UGC-VideoCap">
</p>

## Benchmark Results

<p align="center">
    <img src="benchmark.png" alt="UGC-VideoCap">
</p>

### Model Zoom

| Model                         | visual | audio | details | average | Link |
|:------------------------------|:------:|:-----:|:-------:|:-------:|:----:|
| Gemini-2.5-pro              |    75.8    |   70.8    |    74.8     |     73.78    | N/A  |
| Gemini-2.5-flash              | **78.8**     |   **74.2**    |    **77.2**    |    **76.73**    | N/A  |
| Qwen2.5-Omni-3B               |   55.6     |  48.2    |   52.6      |   52.18      | N/A  |
| UGC-VideoCaptioner-3B-zero(1k RL)         |    57.8    |  53.0     |    55.4     |    55.40(**+3.22**)    | [google-drive](https://drive.google.com/drive/folders/1R-L4kz4R7UxYpcU4El1ctgvVDQbsMsG6?usp=sharing) |
| Qwen2.5-Omni-3B 1k sft        |    58.4    |   61.4  |   57.0    |     58.96(+6.78)    | [google-drive](https://drive.google.com/drive/folders/1itJ1u4XEJNVfmgbxuKL-fGWCbaW3EAza?usp=sharing) |
| Qwen2.5-Omni-3B 10k sft       |    58.4   |    63.2   |   58.0     |   59.87(+7.69)     | [google-drive](https://drive.google.com/drive/folders/1auQ4mx9CcxIzAIF4SyH034xufzrqe29w?usp=sharing) |
| Qwen2.5-Omni-3B 20k sft       |    59.2   |   64    |    58.4   |     60.50(+8.32)     | [google-drive](https://drive.google.com/drive/folders/11WJZkq8I_807zJUmBCCvwNjSj18F2im9?usp=sharing) |
| UGC-VideoCaptioner-3B (1k SFT + 1k RL)         |   59.4     |    62.4   |    58.2     |    60.01(**+7.83**)   | [google-drive](https://drive.google.com/drive/folders/1LGmIU60cdacErNgUk86D8I5_kiU_ljFz?usp=sharing) |

## Quick Start

You can use this model with the `transformers` library. Below is a quick example demonstrating how to perform inference.
Please note that for full video processing capabilities, you might need to install `decord` and refer to the [official GitHub repository](https://github.com/openinterx/UGC-VideoCaptioner) for detailed video handling steps, especially if `AutoProcessor` doesn't directly handle video file paths for complex scenarios.

### Environment Setup

```bash
pip install transformers torch decord soundfile qwen_omni_utils
```

### Inference

```python
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

model = Qwen2_5OmniForConditionalGeneration.from_pretrained("openinterx/UGC-VideoCaptioner", torch_dtype="auto", device_map="auto")

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-Omni-3B",
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )

processor = Qwen2_5OmniProcessor.from_pretrained("openinterx/UGC-VideoCaptioner")

# Example video path (replace with your actual video file path)
video_path = "path/to/your/video.mp4" 

# Define the detailed captioning prompt
prompt_text = (
    "You are given a short video with both audio and visual content. Write a detailed and coherent paragraph "
    "that naturally integrates all modalities. Your description should include: (1) the primary scene and "
    "background setting; (2) key characters or objects and their actions or interactions; (3) significant "
    "audio cues such as voices, background music, sound effects, and their emotional tone; (4) any on-screen "
    "text (OCR) and its role in the video context; and (5) the overall theme or purpose of the video. "
    "Ensure the output is a fluent and objective paragraph, not a bullet-point list, and captures the video's "
    "content in a human-like, narrative style."
)

# Prepare messages in the chat template format
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path}, # Pass video path
            {"type": "text", "text": prompt_text},
        ],
    }
]


# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
```



### vllm inference 

```python
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

```


## Evaluation

### Final Caption prompt (for inference)

```python
prompt = "You are given a short video with both audio and visual content. Write a detailed and coherent paragraph that naturally integrates all modalities. "
"Your description should include: (1) the primary scene and background setting; (2) key characters or objects and their actions or interactions; "
"(3) significant audio cues such as voices, background music, sound effects, and their emotional tone; "
"(4) any on-screen text (OCR) and its role in the video context; and (5) the overall theme or purpose of the video. "
"Ensure the output is a fluent and objective paragraph, not a bullet-point list, and captures the video's content in a human-like, narrative style.
```

### Score

Scores are judged by GPT-4o-2024-08-06.

```bash
python eval_caption.py
```

## Citation

If you find this repository helpful, feel free to cite our paper:

```bibtex
@article{wu2024ugc,
  title={UGC-VideoCaptioner: An Omni UGC Video Detail Caption Model and New Benchmarks},
  author={Wu, Zhenyu and Sun, Qiushi and Zhang, Jiabo and Zhu, Yuyin and Ma, Guojun and Cheng, Kanzhi and Jia, Chengyou and Tan, Jian and Yang, Qing and Wu, Zhiyong},
  journal={arXiv preprint arXiv:2507.11336},
  year={2024}
}
```