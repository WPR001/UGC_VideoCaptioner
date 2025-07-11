
import json
import openai
from tqdm import tqdm

# === 配置项 ===
INPUT_JSON = "minicpm_output.json"
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

# === System Prompt ===
system_prompt = (
    "You are given two object lists from a video: one is the ground truth (`answer`), "
    "and the other is a predicted list (`predict`). Each object is described in the format "
    "'object (count: X, color: Y, position: Z)'.\n\n"
    "Your task is to compute how well the predicted list covers the objects in the answer list.\n\n"
    "Please follow these rules:\n"
    "1. Only consider the object entries in the `answer` list.\n"
    "2. For each object, check whether the predicted list contains an object with matching or semantically equivalent:\n"
    "   - category (object name)\n"
    "   - count\n"
    "   - color\n"
    "   - position\n"
    "3. Ignore only formatting differences (e.g., line breaks, punctuation, or capitalization).\n"
    "4. Extra objects in the predicted list do NOT affect the score.\n\n"
    "The final score is: matched_objects / total_objects_in_answer (e.g., 2/3 = 0.67).\n\n"
    "Return ONLY a JSON object in the following format:\n"
    "{\"match_score\": 0.xx}"
)

# === 单样本评分函数 ===
def call_match_score_api(video_id, answer, predict):
    user_prompt = (
        f"Sample ID: {video_id}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Predict:\n{predict}\n\n"
        "IMPORTANT: Only return a plain JSON object like:\n"
        "{\"match_score\": 0.67}\n"
        "Do not add any explanation, comment, or markdown formatting such as ```json."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        reply = response.choices[0].message.content.strip()

        # 去除 markdown 包裹内容（如 ```json ... ```）
        if reply.startswith("```"):
            reply = reply.strip().strip("```").strip("json").strip()

        try:
            result = json.loads(reply)
            return float(result.get("match_score", 0.0))
        except json.JSONDecodeError:
            print(f"[PARSE ERROR] {video_id}: GPT reply not valid JSON.\nReply was:\n{reply}\n")
            return 0.0

    except Exception as e:
        print(f"[API ERROR] {video_id}: {e}")
        return 0.0

# === 主评估逻辑 ===
def evaluate_match_score(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_scores = []

    for item in tqdm(data["samples"], desc="Evaluating"):
        video_id = item.get("video_id")
        answer = (item.get("answer") or "").strip()
        predict = (item.get("predict") or "").strip()

        if not answer or not predict:
            continue  # 跳过空样本

        score = call_match_score_api(video_id, answer, predict)
        all_scores.append(score)

        print(f"{video_id} => Score: {score:.2f}")

    # === 平均得分 ===
    if all_scores:
        average_score = sum(all_scores) / len(all_scores)
        print(f"\n✅ Final Average Match Score: {average_score:.2%} over {len(all_scores)} samples")
    else:
        print("⚠️ No valid samples found.")

# === 执行入口 ===
if __name__ == "__main__":
    evaluate_match_score(INPUT_JSON)