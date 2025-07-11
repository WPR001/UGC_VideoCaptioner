import json
import openai
from tqdm import tqdm

# === 配置项 ===
# INPUT_JSON = "gemini_2.5_flash.json"  # Final Accuracy: 53.55% (166/310)
INPUT_JSON = "minicpm_output.json"  # Final Accuracy: 51.30%
OPENAI_API_KEY = ""

openai.api_key = OPENAI_API_KEY

# === 工具函数：提取GPT判断是否一致 ===
def judge_background_change_consistency(video_id, answer, predict):
    system_prompt = (
        "You are given a ground truth video background description and a predicted background description. "
        "Determine whether they describe the same pattern of background change. This includes:\n"
        "1. Whether both descriptions agree on whether the background changes.\n"
        "2. Whether the described scenes are consistent in terms of location transitions.\n\n"
        "Return ONLY a JSON object like this: {\"consistent\": \"yes\"} or {\"consistent\": \"no\"}\n\n"
        "Be strict: hallucinations, invented scenes, or wrong transitions count as 'no'."
    )

    user_prompt = (
        f"Sample ID: {video_id}\n\n"
        f"Ground Truth (answer): {answer}\n\n"
        f"Prediction (predict): {predict}\n"
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
        reply = response.choices[0].message.content
        match = json.loads(reply)
        return match.get("consistent", "").lower() == "yes"
    except Exception as e:
        print(f"[ERROR] {video_id}: {e}")
        return False  # 保守地认为不一致

# === 主函数 ===
def evaluate_consistency(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = 0
    correct = 0

    for item in tqdm(data["samples"], desc="Evaluating"):
        video_id = item.get("video_id")
        answer = (item.get("answer") or "").strip()
        predict = (item.get("predict") or "").strip()

        # 跳过空样本，不计入 total
        if not answer or not predict:
            continue

        is_consistent = judge_background_change_consistency(video_id, answer, predict)

        total += 1
        if is_consistent:
            correct += 1

        print(f"{video_id} => {'✓' if is_consistent else '✗'}")

    if total > 0:
        acc = correct / total
        print(f"\n✅ Final Accuracy: {acc:.2%} ({correct}/{total})")
    else:
        print("⚠️ No valid samples found.")

# === 执行 ===
if __name__ == "__main__":
    evaluate_consistency(INPUT_JSON)