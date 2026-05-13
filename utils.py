import streamlit as st
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
import torch
import whisper

# ---------- Emotion-to-sentiment mapping ----------
emotion_to_sentiment = {
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sad": "negative",
    "happy": "positive",
    "surprise": "positive",
    "neutral": "neutral"
}
label_order_image = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
label_order_text = ["positive", "negative", "neutral"]

@st.cache_resource
def load_image_model():
    """ViT for facial emotions with eager attention → enables attention heatmaps."""
    model_name = "dima806/facial_emotions_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        attn_implementation="eager"      # ← forces standard attention, returns weights
    )
    return processor, model

@st.cache_resource
def load_text_model():
    """RoBERTa sentiment (3‑class) with eager attention for token visualisation."""
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        attn_implementation="eager"
    )
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )

@st.cache_resource
def load_whisper():
    """Whisper base for speech‑to‑text."""
    return whisper.load_model("base")