import json
import re
from tqdm import tqdm
from rouge_score import rouge_scorer

INPUT_JSONL = "minicpm_ocr_output_cleaned.jsonl"

def simple_tokenize(text):
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\\n|\\t|\n|\t', ' ', text)
    text = re.sub(r'[^\w\s\'\"]+', '', text)  # 保留字母、数字、空格、撇号、引号
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_from_raw_string(s):
    # 从 "abc", "XYZ", "You Lose!" 这种格式中提取
    return re.findall(r'"([^"]+)"', s)

def try_parse_ocr_list(data):
    """统一解析 ocr 和 predict_ocr["ocr"] 内容"""
    if isinstance(data, list):
        return [normalize_text(x) for x in data if isinstance(x, str)]
    if isinstance(data, str):
        data = data.strip()
        if data.startswith("```json"):
            data = data[7:]
        elif data.startswith("```"):
            data = data[3:]
        if data.endswith("```"):
            data = data[:-3]

        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and "ocr" in parsed:
                return try_parse_ocr_list(parsed["ocr"])
            elif isinstance(parsed, list):
                return try_parse_ocr_list(parsed)
        except Exception:
            # fallback: 正则提取 "..." 中的内容
            return [normalize_text(x) for x in extract_from_raw_string(data)]

    if isinstance(data, dict) and "ocr" in data:
        return try_parse_ocr_list(data["ocr"])

    return [normalize_text(str(data))]

def evaluate_ocr_nlp(input_path):
    scores = {
        'bleu1': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Evaluating OCR with NLP metrics"):
            item = json.loads(line)
            vid = item.get("video_id", "<no-id>")
            gt_text_list = try_parse_ocr_list(item.get("ocr", []))
            pred_text_list = try_parse_ocr_list(item.get("predict_ocr", {}))

            if not gt_text_list or not pred_text_list:
                print(f"[{vid}] skip empty ocr")
                continue

            gt_text = " ".join(gt_text_list)
            pred_text = " ".join(pred_text_list)

            ref_tokens = [simple_tokenize(gt_text)]
            hyp_tokens = simple_tokenize(pred_text)

            # BLEU-1（简单重叠率）
            ref_counts = {}
            for tok in ref_tokens[0]:
                ref_counts[tok] = ref_counts.get(tok, 0) + 1
            match = 0
            for tok in hyp_tokens:
                if ref_counts.get(tok, 0) > 0:
                    match += 1
                    ref_counts[tok] -= 1
            bleu1 = match / len(hyp_tokens) if hyp_tokens else 0.0
            scores['bleu1'].append(bleu1)

            # ROUGE
            rouge_scores = scorer.score(gt_text, pred_text)
            r1 = rouge_scores['rouge1'].fmeasure
            r2 = rouge_scores['rouge2'].fmeasure
            rL = rouge_scores['rougeL'].fmeasure
            scores['rouge1'].append(r1)
            scores['rouge2'].append(r2)
            scores['rougeL'].append(rL)

            print(
                f"[{vid}] BLEU-1: {bleu1:.4f}, "
                f"ROUGE-1: {r1:.4f}, ROUGE-2: {r2:.4f}, ROUGE-L: {rL:.4f}"
            )

    avg = {m: (sum(v)/len(v) if v else 0.0) for m, v in scores.items()}
    overall = sum(avg.values()) / len(avg)

    print("\n✅ 平均指标（英文，不区分大小写）：")
    print(f"BLEU-1   : {avg['bleu1']:.4f}")
    print(f"ROUGE-1  : {avg['rouge1']:.4f}")
    print(f"ROUGE-2  : {avg['rouge2']:.4f}")
    print(f"ROUGE-L  : {avg['rougeL']:.4f}")
    print(f"\n🎯 综合 Accuracy: {overall:.4f}")

if __name__ == "__main__":
    evaluate_ocr_nlp(INPUT_JSONL)