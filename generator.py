def generate_summary(vis_emotion, vis_conf, txt_sentiment, txt_conf, fusion_result):
    """Generate a rich natural-language summary of the multimodal emotional assessment."""
    vis_pct = f"{vis_conf*100:.0f}%"
    txt_pct = f"{txt_conf*100:.0f}%"

    emotion_descriptors = {
        "happy":    "joy and positivity",
        "sad":      "sadness or low mood",
        "angry":    "frustration or anger",
        "fear":     "anxiety or fear",
        "disgust":  "discomfort or aversion",
        "surprise": "surprise or astonishment",
        "neutral":  "a composed, neutral affect",
    }
    sentiment_descriptors = {
        "positive": "optimistic and affirming",
        "negative": "critical or pessimistic",
        "neutral":  "measured and impartial",
    }

    vis_desc = emotion_descriptors.get(vis_emotion.lower(), vis_emotion)
    txt_desc = sentiment_descriptors.get(txt_sentiment.lower(), txt_sentiment)

    base = (
        f"The facial expression analysis (ViT model, {vis_pct} confidence) detects <strong>{vis_emotion}</strong>, "
        f"indicating {vis_desc}. "
        f"The language model (RoBERTa, {txt_pct} confidence) classifies the text as <strong>{txt_sentiment}</strong> — "
        f"a tone that is {txt_desc}. "
    )

    if fusion_result == "MISMATCH":
        detail = (
            "The fusion layer identifies a <strong>significant mismatch</strong> between these two signals. "
            "Despite expressing verbally positive or neutral language, the speaker's facial cues suggest underlying stress, "
            "discomfort, or emotional suppression. "
            "This incongruence is a well-documented phenomenon in affective computing and may indicate "
            "that the individual is masking their true emotional state. "
            "This pattern warrants careful attention in interpersonal contexts."
        )
    else:
        detail = (
            "The fusion layer confirms that both modalities are <strong>aligned</strong>. "
            "The speaker's facial expression and verbal content convey a consistent emotional message, "
            "suggesting authenticity between what is felt and what is communicated. "
            "No emotional suppression or masking is detected."
        )

    return base + detail
