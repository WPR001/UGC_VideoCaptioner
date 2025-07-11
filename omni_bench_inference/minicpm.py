import os
import json
import math
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import librosa
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    duration = int(video.duration)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000, verbose=False, logger=None)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)

    # === 每秒一帧（1fps）
    timestamps = list(range(duration))

    # 若超过32帧，则等间隔采样32个索引（含首尾）
    if len(timestamps) > 32:
        indices = np.linspace(0, len(timestamps) - 1, 32, dtype=int).tolist()
        timestamps = [timestamps[i] for i in indices]

    contents = []
    for t in timestamps:
        try:
            frame = video.get_frame(t)
            image = Image.fromarray(frame.astype(np.uint8))
            audio = audio_np[sr * t: sr * (t + 1)]
            if flatten:
                contents.extend(["<unit>", image, audio])
            else:
                contents.append(["<unit>", image, audio])
        except Exception as e:
            print(f"[Warning] Error at second {t} in {video_path}: {e}")
            continue

    return contents


def generate_caption(model, tokenizer, video_path):
    contents = get_video_chunk_content(video_path)
    sys_msg = model.get_sys_prompt(mode='omni', language='en')
    msg = {"role": "user", "content": contents}
    msgs = [sys_msg, msg]

    res = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.5,
        max_new_tokens=1024,
        omni_input=True,
        use_tts_template=False,
        generate_audio=False,
        max_slice_nums=1,
        use_image_id=False,
        return_dict=True
    )
    return res['text']


def main(video_folder, output_jsonl):
    print("Loading model...")
    model = AutoModel.from_pretrained(
        'openbmb/MiniCPM-o-2_6',
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=True
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

    video_exts = (".mp4", ".mov", ".avi", ".mkv")

    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for fname in tqdm(os.listdir(video_folder)):
            if not fname.lower().endswith(video_exts):
                continue
            video_path = os.path.join(video_folder, fname)
            video_id = os.path.splitext(fname)[0]

            try:
                caption = generate_caption(model, tokenizer, video_path)
                fout.write(json.dumps({"video_id": video_id, "caption": caption}, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Error] Failed to process {video_id}: {e}")


if __name__ == "__main__":
    # 修改为你的视频路径和输出文件名
    video_folder = "/path/to/your/video/folder"
    output_jsonl = "minicpm_captions.jsonl"
    main(video_folder, output_jsonl)