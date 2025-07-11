import json
import re
from tqdm import tqdm
from rouge_score import rouge_scorer

INPUT_JSONL = "gemma_e2b_ocr_output.jsonl"

def simple_tokenize(text):
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)

def clean_predict_text(text):
    """去除所有符号、标点、控制字符、字段名等，仅保留纯英文和数字内容"""
    text = text.lower()
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = text.replace('```json', '').replace('```', '')
    text = re.sub(r'\\+', '', text)                     # 去除反斜杠
    text = re.sub(r'\"', '', text)                      # 去除引号
    text = re.sub(r'\[|\]|\{|\}', '', text)             # 去除括号
    text = re.sub(r'\bocr\b\s*:', '', text)             # 去除 ocr: 字段名
    text = re.sub(r'[^\w\s\']+', '', text)              # 去除除撇号外的标点
    text = re.sub(r'\s+', ' ', text).strip()            # 去除多余空格
    return text

def try_parse_predict_ocr(pred_field):
    if isinstance(pred_field, dict):
        return clean_predict_text(" ".join(pred_field.get("ocr", [])))

    if isinstance(pred_field, str):
        cleaned = pred_field.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "ocr" in parsed:
                return clean_predict_text(" ".join(parsed["ocr"]))
            elif isinstance(parsed, list):
                return clean_predict_text(" ".join(parsed))
        except Exception:
            pass

        return clean_predict_text(cleaned)

    return clean_predict_text(str(pred_field))

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
            gt_segments = item.get("ocr", [])
            pred_field = item.get("predict_ocr", None)

            if not gt_segments:
                print(f"[{vid}] skip empty ground truth")
                continue

            gt_text = " ".join(gt_segments).lower()
            pred_text = try_parse_predict_ocr(pred_field)

            ref_tokens = [simple_tokenize(gt_text)]
            hyp_tokens = simple_tokenize(pred_text)

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