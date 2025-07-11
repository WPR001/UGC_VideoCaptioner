import os
import re
import json
from datetime import datetime
from tqdm import tqdm
import openai


INPUT_JSONL = ""
JSON_FOLDER = ""
OUTPUT_JSON = ""
OPENAI_API_KEY = ""

def safe_parse_evaluation(response_str):
    try:
        match = re.search(r"\{.*\}", response_str, re.DOTALL)
        if match:
            return json.loads(match.group(0).replace("'", '"'))
    except Exception as e:
        print(f"Error parsing evaluation output: {e}")
    return {}

def evaluate_caption(sample_id, pred_caption, true_caption, api_key):
    openai.api_key = api_key

    system_msg = (
        "You are an assistant that compares a ground truth video description and a predicted video description. "
        "Evaluate the predicted description against the ground truth on the following three dimensions:\n"
        "1. **visual**: the accuracy and completeness of visual content including the scene setting, background, characters or objects, their actions or interactions, and any OCR text.\n"
        "2. **audio**: how well it captures voices, background music, sound effects, and their emotional tone.\n"
        "3. **details**: the completeness, thematic consistency, purpose, coherence, and integration of multimodal content.\n\n"

        "For each dimension, assign an integer score from 1 to 5, following these detailed grading criteria:\n\n"

        "**Score 1:** The description is mostly irrelevant or misleading. It misrepresents or omits most key information. "
        "At least 3 important elements are missing or incorrect. Severe hallucinations may be present.\n\n"

        "**Score 2:** The description captures a few elements (1-2 aspects) but is vague or inaccurate for the rest. "
        "It is poorly structured or confusing, with major omissions or incorrect details.\n\n"

        "**Score 3:** The description aligns with the video on most elements (3 or more), but lacks depth or specificity. "
        "Some key details are missing, or minor factual errors exist. It's generally correct but too generic or incomplete.\n\n"

        "**Score 4:** A mostly accurate and complete description. Captures nearly all key information (4+ aspects), "
        "with clear structure and appropriate level of detail. Minor omissions or simplifications are acceptable.\n\n"

        "**Score 5:** Exceptionally accurate and detailed. Covers all relevant aspects thoroughly, with well-integrated information. "
        "Captures subtle nuances (e.g., emotion, scene dynamics, audio-visual interplay) and reads like it was written by a domain expert.\n\n"


        "Respond only with a valid Python dictionary in this format:\n"
        "{'visual': int, 'audio': int, 'details': int}"
    )

    user_msg = (
        f"Sample ID: {sample_id}\n"
        f"Predicted Description: {pred_caption}\n"
        f"Ground Truth Description: {true_caption}\n"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",
            temperature=1,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        )
        content = response.choices[0].message.content
        eval_dict = safe_parse_evaluation(content)
        v = int(eval_dict.get('visual', 0))
        a = int(eval_dict.get('audio', 0))
        d = int(eval_dict.get('details', 0))
    except Exception as e:
        print(f"Error evaluating sample {sample_id}: {e}")
        v = a = d = 0

    return {'visual': v, 'audio': a, 'details': d}

def run_evaluation():
    results = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                results = existing.get('evaluations', {})
        except Exception as e:
            print(f"Warning: Failed to read previous results: {e}")
            results = {}

    sum_avg = 0.0
    sum_visual = 0.0
    sum_audio = 0.0
    sum_details = 0.0
    count = 0

    for v in results.values():
        sum_avg += v.get('average_score', 0.0)
        sum_visual += v.get('visual_score', 0.0)
        sum_audio += v.get('audio_score', 0.0)
        sum_details += v.get('details_score', 0.0)
        count += 1

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading samples"):
            try:
                sample = json.loads(line)
                video_id = sample.get('video_id')
                pred_caption = sample.get('caption', '').strip()
                if not video_id or not pred_caption:
                    continue
                if video_id in results:
                    continue

                json_path = os.path.join(JSON_FOLDER, f"{video_id}.json")
                if not os.path.isfile(json_path):
                    continue

                with open(json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                true_caption = data.get('final_caption', {}).get('caption_text', '').strip()
                if not true_caption:
                    continue

                scores = evaluate_caption(video_id, pred_caption, true_caption, OPENAI_API_KEY)
                avg_score = sum(scores.values()) / len(scores)

                sum_avg += avg_score
                sum_visual += scores['visual']
                sum_audio += scores['audio']
                sum_details += scores['details']
                count += 1

                results[video_id] = {
                    'visual_score': scores['visual'],
                    'audio_score': scores['audio'],
                    'details_score': scores['details'],
                    'average_score': avg_score
                }

                overall_percent = (sum_avg / count) / 5.0 * 100
                current = {
                    'evaluations': results,
                    'overall_average_percent': overall_percent,
                    'overall_visual_average': sum_visual / count,
                    'overall_audio_average': sum_audio / count,
                    'overall_details_average': sum_details / count
                }
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as out_f:
                    json.dump(current, out_f, indent=2, ensure_ascii=False)

                print(
                    f"Processed {video_id}: "
                    f"visual={scores['visual']}, audio={scores['audio']}, details={scores['details']}, "
                    f"avg={avg_score:.2f}, overall%={overall_percent:.2f}"
                )

            except json.JSONDecodeError:
                continue

    if count > 0:
        final_output = {
            'evaluations': results,
            'overall_average_percent': (sum_avg / count) / 5.0 * 100,
            'overall_visual_average': sum_visual / count,
            'overall_audio_average': sum_audio / count,
            'overall_details_average': sum_details / count
        }
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as out_f:
            json.dump(final_output, out_f, indent=2, ensure_ascii=False)
        print("Evaluation complete.")
        print(f"Overall visual avg: {final_output['overall_visual_average']:.2f}, "
              f"audio avg: {final_output['overall_audio_average']:.2f}, "
              f"details avg: {final_output['overall_details_average']:.2f}, "
              f"overall%: {final_output['overall_average_percent']:.2f}")

if __name__ == '__main__':
    run_evaluation()