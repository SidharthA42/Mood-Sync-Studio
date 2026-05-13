import re

import numpy as np
import torch

from utils import label_order_text, load_text_model


def _split_clauses(text: str):
    """Split text into clauses on punctuation and contrast words."""
    parts = re.split(
        r"(?<=[.!?])\s+|(?:\s+but\s+|\s+however\s+|\s+although\s+|\s+though\s+|\s+yet\s+|\s+while\s+|\s+whereas\s+)",
        text,
        flags=re.IGNORECASE,
    )
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
    return parts if parts else [text]


def _merge_subword_attention(tokens, weights):
    """Merge RoBERTa subword pieces so words like 'okay' plot as one bar."""
    merged = []
    current_token = ""
    current_weights = []

    for token, weight in zip(tokens, weights):
        if token in ["<s>", "</s>", "<pad>", "Ä "]:
            continue

        starts_new = token.startswith("\u0120") or token.startswith("Ġ") or token.startswith("Ä ")
        clean = token.replace("\u0120", "").replace("Ġ", "").replace("Ä ", "").strip()
        if not clean:
            continue

        if starts_new and current_token:
            merged.append((current_token, float(max(current_weights))))
            current_token = clean
            current_weights = [float(weight)]
        else:
            current_token += clean
            current_weights.append(float(weight))

    if current_token:
        merged.append((current_token, float(max(current_weights))))

    return merged


def predict_sentiment(text: str, return_attention=False):
    pipe = load_text_model()

    segments = _split_clauses(text)
    all_scores = {label: [] for label in label_order_text}

    for segment in segments:
        raw = pipe(segment)
        inner = raw[0] if isinstance(raw[0], list) else raw
        for item in inner:
            if isinstance(item, dict) and "label" in item and "score" in item:
                label = item["label"]
                if label in all_scores:
                    all_scores[label].append(item["score"])

    avg_scores = {label: float(np.mean(scores)) if scores else 0.0 for label, scores in all_scores.items()}
    total = sum(avg_scores.values())
    probs_dict = {label: avg_scores[label] / total for label in label_order_text} if total > 0 else {label: 0.0 for label in label_order_text}

    top_label = max(probs_dict, key=probs_dict.get)
    top_conf = probs_dict[top_label]

    attention_tokens = None
    if return_attention:
        try:
            tokenizer = pipe.tokenizer
            model = pipe.model
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
            last_attn = outputs.attentions[-1]
            avg_attn = last_attn.mean(dim=1).squeeze(0)
            cls_attn = avg_attn[0, :].cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            attention_tokens = _merge_subword_attention(tokens, cls_attn)
        except Exception as exc:
            print(f"Attention extraction error: {exc}")

    return top_label, top_conf, probs_dict, attention_tokens
